"""One map presentation across selection, staged edits and concrete dispatch."""
from copy import deepcopy

from .policies import PolicyViolation
from .provider_budget import effective_request_body
from .retrieval.snapshot import snapshot_plan, render_view, provisional_view, attach_view

_COUNTED_CODES = frozenset(('within_normal_budget', 'reserve_used', 'temporary_headroom',
    'provider_window_exceeded', 'reserve_reason_required', 'unclassified_headroom', 'return_plan_required'))


def replace_messages(request, messages):
    out = deepcopy(request)
    extra = out.get('extra_body')
    if type(extra) is dict and 'messages' in extra:
        extra['messages'] = messages
    else:
        out['messages'] = messages
    return out


def require_count(decision):
    if decision.get('budget_mode') != 'exact' or decision.get('code') not in _COUNTED_CODES:
        raise PolicyViolation(decision['code'], decision['message'],
            required_tokens=decision.get('required_tokens'), available_tokens=decision.get('available_tokens'))
    return decision['prompt_tokens']


class MapSnapshotMixin:
    def _map_view_for(self, store, view=None):
        plan = snapshot_plan(store)
        if plan is None:
            return None
        view = view if view is not None else getattr(self, '_context_map_view', None)
        return deepcopy(view) if view and view['key'] == plan['key'] else provisional_view(plan)

    def _attach_context_map(self, messages, store=None):
        return attach_view(messages, self._map_view_for(store or self.store))

    def _render_map_messages(self, messages):
        return self._request_renderer(messages) if self._request_renderer is not None else messages

    def _measure_context_map(self, request, without, route, *, counted=None, base_count=None, allow_pending=False):
        counted = counted or self._count_budget_request(request, route, allow_pending=allow_pending)
        base_count = base_count or self._count_budget_request(without, route, allow_pending=allow_pending)
        with_tokens, without_tokens = require_count(counted), require_count(base_count)
        delta = with_tokens - without_tokens
        if delta < 0:
            raise PolicyViolation('context_map_count_nonmonotonic', 'Map contribution cannot be certified for this template')
        return dict(snapshot_tokens=delta, with_prompt_tokens=with_tokens, without_prompt_tokens=without_tokens,
            request_fingerprint=counted['prompt_fingerprint'], without_fingerprint=base_count['prompt_fingerprint'],
            count_scope='complete_request_difference', target_tokens=self.config.context_map_token_target,
            cap_tokens=self.config.context_map_token_cap)

    def _require_map_cap(self, measurement):
        if measurement['snapshot_tokens'] > self.config.context_map_token_cap:
            raise PolicyViolation('context_map_cap_exceeded', 'Mandatory framed Context Map exceeds its token cap',
                required_tokens=measurement['snapshot_tokens'], available_tokens=self.config.context_map_token_cap)

    def _fit_context_map(self, request, route, base_messages):
        plan = snapshot_plan(self.store)
        if plan is None:
            return replace_messages(request, self._render_map_messages(base_messages)), None, None
        without = replace_messages(request, self._render_map_messages(base_messages))
        base_count = self._count_budget_request(without, route)
        require_count(base_count)
        ids = [plan['data']['root_id']]

        def measure(chosen):
            view = render_view(plan, chosen)
            candidate = replace_messages(request, self._render_map_messages(attach_view(base_messages, view)))
            measurement = self._measure_context_map(candidate, without, route, base_count=base_count)
            return candidate, view, measurement

        accepted = measure(ids)
        self._require_map_cap(accepted[2])
        for unit in plan['units']:
            proposed = ids + [uid for uid in unit if uid not in ids]
            if proposed == ids:
                continue
            candidate = measure(proposed)
            if candidate[2]['snapshot_tokens'] <= self.config.context_map_token_target:
                accepted, ids = candidate, proposed
        return accepted

    def prepare_context_request(self, request, *, route, purpose='task', renderer=None):
        """Explicit preparation, before the separate read-only send validator."""
        pressure = self.config.budget_mode == 'exact' and self.config.emit_nudges
        if purpose != 'task' or (not self.config.context_map_enabled and not pressure):
            return request
        if self._pending_direct_rollback is not None or self._pending_tool_budget is not None:
            # The final validator owns recovery/abort and refuses this dispatch.
            # Preparation must not intercept that existing ownership boundary.
            return request
        self._require_origin()
        if self.config.context_map_enabled:
            self.config.validate_context_map()
        if renderer is not None:
            self.bind_context_request_renderer(renderer)
        if self.config.budget_mode != 'exact':
            return request
        fit_map = self.config.context_map_enabled and snapshot_plan(self.store) is not None
        if not fit_map and not pressure:
            return request
        if effective_request_body(request).get('messages') != self._render_budget_messages(self.store):
            raise PolicyViolation('context_map_frame_mismatch', 'Final messages do not match the selected map presentation')
        old_notice = deepcopy(self._exact_pressure_notice)
        old_view = deepcopy(getattr(self, '_context_map_view', None))
        candidate = request
        try:
            def fit_current_map(value):
                if not fit_map:
                    return value
                value, view, _ = self._fit_context_map(value, route, self._base_budget_messages(self.store))
                self._context_map_view = view
                return value

            candidate = fit_current_map(candidate)
            if not pressure:
                return candidate
            from .prompts import build_prepared_notice, build_exact_recovery_notice
            limits = self._exact_limits()
            usable = limits.normal_threshold
            highest_level = None
            recovery_band = 0
            # Within one preparation advice may only escalate. This prevents
            # oscillation when the notice itself straddles a threshold. Every
            # changed rendering is counted again; no pressure episode is spent.
            for _ in range(len(self.config.nudge_thresholds) + 4):
                decision = self._count_budget_request(candidate, route)
                tokens = require_count(decision)
                crossed = [level for level in self.config.nudge_thresholds if tokens >= level * usable]
                if crossed:
                    highest_level = max(max(crossed), highest_level or 0)
                notice = None if highest_level is None else {
                    'level': highest_level, 'usable_tokens': usable,
                    'text': build_prepared_notice(highest_level, usable),
                }
                # Stable bands also distinguish reserve from above-T headroom.
                # Reading the episode does not consume or authorize an attempt.
                recovery_band = max(recovery_band,
                    2 if tokens > limits.active_target else 1 if tokens > usable else 0)
                if recovery_band:
                    # Bounded recovery turn: arm on the measured crossing; the
                    # directive rides this preparation's recovery notice.
                    self._maybe_arm_recovery_turn(tokens, usable)
                    episode = self._pressure_episode
                    notice = {
                        'level': max(highest_level or 0, 1.0),
                        'usable_tokens': usable, 'recovery_band': recovery_band,
                        'text': build_exact_recovery_notice(
                            usable_tokens=usable, active_target=limits.active_target,
                            above_target=recovery_band == 2, attempts=episode.attempts,
                            max_attempts=episode.max_attempts, nonprogress=episode.nonprogress,
                            max_nonprogress=episode.max_nonprogress)
                            + self._recovery_directive_suffix(),
                        'tail_echo': self._recovery_tail_echo() or '',
                    }
                if notice == self._exact_pressure_notice:
                    if notice != old_notice and notice is not None:
                        self.telemetry.record('nudge', initiator='host', level=notice['level'],
                            active_tokens=tokens, budget_mode='exact',
                            count_scope='complete_current_request_preparation')
                    return candidate
                self._exact_pressure_notice = notice
                candidate = replace_messages(candidate, self._render_budget_messages(self.store))
                candidate = fit_current_map(candidate)
            raise PolicyViolation('pressure_notice_preparation_exhausted',
                                  'Pressure notice did not stabilize within its preparation bound')
        except BaseException:
            self._exact_pressure_notice = old_notice
            self._context_map_view = old_view
            raise

    def _final_context_map_measurement(self, request, route, counted):
        if snapshot_plan(self.store) is None:
            return None
        if effective_request_body(request).get('messages') != self._render_budget_messages(self.store):
            raise PolicyViolation('context_map_frame_mismatch', 'Final messages do not match the selected map presentation')
        without = replace_messages(request, self._render_map_messages(self._base_budget_messages(self.store)))
        result = self._measure_context_map(request, without, route, counted=counted)
        self._require_map_cap(result)
        return result
