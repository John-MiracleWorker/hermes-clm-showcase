"""CLM's consumer of the generic final-request enforcement contract."""
from copy import deepcopy
from contextlib import contextmanager

from .provider_budget import (BudgetLimits, PressureEpisode, ProviderIdentity,
                              enforce_final_request, effective_request_body)
from .provider_tokenizer import CounterError
from .policies import PolicyViolation


class ExactBudgetMixin:
    def _initialize_exact_budget(self):
        self.config.validate_budget()
        self._exact_counter = None
        self._request_renderer = None
        self._budget_identity = None
        self._budget_binding_invalid = False
        self._budget_frame = None
        self._last_budget_decision = None
        self._exact_pressure_notice = None
        self._pressure_episode = PressureEpisode()
        self._budget_excursion = None
        self._budget_operation = None
        self.store._budget_validator = self._validate_store_budget

    def bind_exact_counter(self, counter, identity):
        """Bind a profile-owned count service; binding does not certify its counts.

        The concrete ExactRequestCounter requires qualified parity records and
        checks live identity on every use. Each returned PromptCount must still
        be exact and match this complete request and provider identity.
        """
        self._require_origin()
        if not isinstance(identity, ProviderIdentity) or not callable(getattr(counter, 'count_final_request', None)):
            raise ValueError('provider identity and final-request counter are required')
        self.config.validate_budget()
        self._exact_counter = counter
        self._budget_identity = identity
        self._budget_binding_invalid = False
        self._budget_frame = None
        self._last_budget_decision = None
        self._exact_pressure_notice = None
        self._pressure_episode.reset()
        self._recovery_turn.reset()
        self._budget_excursion = None
        self.context_length = identity.context_tokens
        self.budget.provider_context_length = identity.context_tokens

    def _exact_limits(self):
        return BudgetLimits(
            provider_ceiling=self._budget_identity.context_tokens,
            active_target=self.config.active_target_tokens,
            reserve=self.config.reserve_tokens,
            output_tokens=self.config.output_tokens,
            safety_margin=self.config.safety_margin_tokens)

    def begin_budget_excursion(self, reason, return_plan):
        """Declare a bounded recovery plan; redeclaration grants no new attempts."""
        self._require_origin()
        if reason not in ('restore', 'management', 'protected_group', 'pending_group'):
            raise ValueError('A specific budget excursion reason is required')
        if type(return_plan) is not str or not return_plan.strip() or len(return_plan) > 2000:
            raise ValueError('A bounded return-below-S plan is required')
        self._budget_excursion = {'reason': reason, 'return_plan': return_plan}

    def _base_budget_messages(self, store):
        if store.coordinate_origin is not None:
            messages = self._attach_clm_guidance(store._validate_native_presentation())
        else:
            from .selector import ContextSelector
            messages = self._attach_clm_guidance(ContextSelector(store, self.budget).build())
        return self._attach_exact_pressure_notice(messages)

    def _refresh_exact_pressure_notice(self):
        """Refresh only on a threshold transition, using a prior exact frame.

        Selection has not yet built the current final provider request. A prior
        verified task-frame count can inform the model, but cannot admit this
        request or grant reserve. The ordinary final guard still owns admission.
        """
        frame = self._budget_frame
        if (self.config.budget_mode != 'exact' or not self.config.emit_nudges
                or frame is None or frame['conversation_id'] != self.store.conversation_id
                or not frame['decision'].get('allowed')):
            self._exact_pressure_notice = None
            return
        tokens = frame['decision']['prompt_tokens']
        usable = self._exact_limits().normal_threshold
        # Bounded recovery turn: host-owned arming follows the measured crossing.
        self._maybe_arm_recovery_turn(tokens, usable)
        suffix = self._recovery_directive_suffix()
        crossed = [level for level in self.config.nudge_thresholds if tokens >= level * usable]
        if not crossed:
            self._exact_pressure_notice = None
            return
        level = max(crossed)
        echo = self._recovery_tail_echo() or ''
        previous = self._exact_pressure_notice
        # Prior evidence stays frozen within a band; only a band transition or a
        # change in the bounded recovery directive rebuilds the notice.
        if (previous and previous['level'] == level
                and previous['usable_tokens'] == usable
                and previous.get('recovery_suffix', '') == suffix
                and previous.get('tail_echo', '') == echo):
            return
        from .prompts import build_exact_notice
        self._exact_pressure_notice = {
            'level': level, 'usable_tokens': usable,
            'text': build_exact_notice(level, tokens, usable) + suffix,
            'recovery_suffix': suffix, 'tail_echo': echo,
        }
        self.telemetry.record('nudge', initiator='host', level=level, active_tokens=tokens,
                              budget_mode='exact', count_scope='previous_verified_task_frame')

    def _pressure_notice_text(self):
        return self._exact_pressure_notice['text'] if self._exact_pressure_notice else None

    def _attach_exact_pressure_notice(self, messages):
        notice = self._pressure_notice_text()
        if self.config.budget_mode == 'exact' and notice:
            prefix = 0
            while prefix < len(messages) and messages[prefix].get('role') in {'system', 'developer'}:
                prefix += 1
            messages.insert(prefix, {'role': 'system', 'content': notice})
        return messages

    def _attach_recovery_tail_echo(self, messages):
        """Tail-visible copy of the bounded-turn directive.

        The prefix notice is an attention sink during tool loops; while the
        bounded recovery turn is offered the directive is echoed after the
        latest result. Exact mode uses the notice-frozen echo (frame equality
        depends on it); the production path builds it from the live turn state.
        """
        if self.config.budget_mode == 'exact':
            echo = (self._exact_pressure_notice or {}).get('tail_echo') if self._exact_pressure_notice else None
        else:
            echo = self._recovery_tail_echo()
        if echo and self._recovery_turn.state == 'offered':
            messages.append({'role': 'system', 'content': echo})
        return messages

    def _prefix_with_current_pressure_notice(self, old_messages, messages, frame):
        """Allow only this engine's counted notice transition in the prefix.

        Ordinary canonical content and every other host addition must still
        match exactly. Ambiguous or displaced notice envelopes refuse reserve.
        The engine-owned tail reminder is excluded on both sides; the caller
        separately requires the current payload to match the trusted render,
        which carries the current reminder at the tail.
        """
        old_notice = frame.get('pressure_notice')
        new_notice = self._pressure_notice_text()
        old_echo = frame.get('tail_echo') or ''
        new_echo = (self._exact_pressure_notice or {}).get('tail_echo') or ''
        if old_notice == new_notice and old_echo == new_echo:
            expected = deepcopy(old_messages) if old_echo else old_messages
        else:
            expected = deepcopy(old_messages)

            def position(items, text):
                envelope = {'role': 'system', 'content': text}
                matches = [i for i, value in enumerate(items) if value == envelope]
                if len(matches) != 1:
                    return None
                index = matches[0]
                if any(type(value) is not dict or value.get('role') not in {'system', 'developer'}
                       for value in items[:index]):
                    return None
                return index

            if old_notice:
                index = position(expected, old_notice)
                if index is None:
                    return None
                expected.pop(index)
            if new_notice:
                index = position(messages, new_notice)
                if index is None or index > len(expected):
                    return None
                expected.insert(index, {'role': 'system', 'content': new_notice})
        if old_echo:
            envelope = {'role': 'system', 'content': old_echo}
            matches = [i for i, value in enumerate(expected) if value == envelope]
            if len(matches) != 1 or matches[0] != len(expected) - 1:
                return None
            expected.pop()
        return expected

    def _source_budget_messages(self, store):
        return self._attach_recovery_tail_echo(
            self._attach_context_map(self._base_budget_messages(store), store))

    def _render_budget_messages(self, store):
        messages = self._source_budget_messages(store)
        return self._request_renderer(messages) if self._request_renderer is not None else messages

    def bind_context_request_renderer(self, renderer):
        if self._pending_direct_rollback is not None:
            # The task guard binds before validation. Preserve the accepted
            # renderer so check_final_request can retry recovery and refuse
            # this request without attaching a new service to the old owner.
            return
        self._require_origin()
        if renderer is not None and not callable(renderer):
            raise ValueError('candidate renderer must be callable or None')
        self._request_renderer = renderer

    def _require_candidate_frame(self, store):
        frame = self._budget_frame
        if frame is None:
            raise PolicyViolation('candidate_frame_missing', 'candidate_frame_missing: Count a complete task request before editing')
        if (frame['conversation_id'] != store.conversation_id
                or frame['source_uids'] != store.order
                or effective_request_body(frame['request']).get('messages') != self._render_budget_messages(store)):
            raise PolicyViolation('candidate_frame_mismatch',
                'candidate_frame_mismatch: The current presentation does not match the counted final request')

    @contextmanager
    def _exact_edit_scope(self, operation):
        if self.config.budget_mode != 'exact':
            yield
            return
        self._require_candidate_frame(self.store)
        previous = self._budget_operation
        self._budget_operation = operation
        self.store._budget_compound_depth += 1
        try:
            yield
        finally:
            self.store._budget_compound_depth -= 1
            self._budget_operation = previous

    def _validate_store_budget(self, store, operation, phase='after'):
        if self.config.budget_mode != 'exact':
            return
        if phase == 'before':
            self._require_candidate_frame(store)
            snapshot = self._transaction_snapshot()
            def rollback_lower(store_preimage):
                # The lower guard owns its pre-protection source snapshot;
                # this callback owns the engine state coupled to admission.
                snapshot['store'] = store_preimage
                self._rollback_direct_edit(snapshot, None)
            return rollback_lower
        if store is not self.store:
            raise PolicyViolation('candidate_store_mismatch', 'Candidate store is not owned by this engine')
        frame = self._budget_frame
        if frame is None:
            raise PolicyViolation('candidate_frame_missing', 'candidate_frame_missing')
        candidate = deepcopy(frame['request'])
        messages = self._render_budget_messages(store)
        # Preserve the SDK's effective extra_body precedence and every other
        # final field, including tools, output allowance and template settings.
        extra = candidate.get('extra_body')
        if type(extra) is dict and 'messages' in extra:
            extra['messages'] = messages
        else:
            candidate['messages'] = messages
        map_measurement = None
        if self.config.context_map_enabled:
            candidate, view, map_measurement = self._fit_context_map(candidate, frame['route'], self._base_budget_messages(store))
            self._context_map_view = view
        excursion = self._budget_excursion or {}
        reason = 'restore' if operation in ('apply_restore', 'mark_active', 'reset_working_set') else excursion.get('reason')
        decision = self._count_budget_request(candidate, frame['route'],
            reason=reason, return_plan=excursion.get('return_plan'))
        if map_measurement is not None:
            decision['context_map'] = map_measurement
        if not decision['allowed']:
            raise PolicyViolation(decision['code'], decision['message'],
                                  required_tokens=decision.get('required_tokens'),
                                  available_tokens=decision.get('available_tokens'))
        if (decision['prompt_tokens'] > self._exact_limits().normal_threshold
                and (self._pressure_episode.nonprogress >= self._pressure_episode.max_nonprogress
                     or self._pressure_episode.attempts >= self._pressure_episode.max_attempts)):
            raise PolicyViolation('pressure_episode_exhausted', 'pressure_episode_exhausted',
                                  required_tokens=decision['prompt_tokens'],
                                  available_tokens=self._exact_limits().normal_threshold)
        before_tokens = frame['decision']['prompt_tokens']
        growth = max(0, decision['prompt_tokens'] - before_tokens)
        cap = self.config.max_restoration_tokens
        if self._pending_tool_budget is None and reason == 'restore' and cap and growth > cap:
            raise PolicyViolation('max_restoration_exceeded',
                f'Rendered restoration adds {growth} tokens; configured limit is {cap}',
                required_tokens=growth, available_tokens=cap)
        self._check_known_tool_tail({**frame, 'request': candidate}, reason, excursion.get('return_plan'))
        if self._pending_tool_budget is None:
            self._pressure_episode.observe_edit(frame['decision']['prompt_tokens'],
                                                decision['prompt_tokens'], self._exact_limits().normal_threshold)
            if decision['budget_class'] == 'normal':
                self._budget_excursion = None
            elif reason == 'restore':
                self._budget_excursion = {'reason': reason, 'return_plan': excursion.get('return_plan')}
        decision['count_scope'] = 'candidate_source_frame'
        self._budget_frame = {**frame, 'request': candidate, 'decision': deepcopy(decision),
                              'pressure_notice': self._pressure_notice_text(),
                              'tail_echo': (self._exact_pressure_notice or {}).get('tail_echo') or None,
                              'context_map_view': deepcopy(getattr(self, '_context_map_view', None))}
        self._last_budget_decision = deepcopy(decision)
        # These are counts of a prospective edit to the last accepted task frame.
        # New assistant/tool tails are counted again at the next actual dispatch.
        for event in store._event_buffer or ():
            if event.get('event') in ('compact', 'compact_body', 'archive', 'restore',
                                      'replace_summary', 'reset_working_set', 'emergency_compaction', 'lookup'):
                event['exact_prompt_before'] = before_tokens
                event['exact_prompt_after'] = decision['prompt_tokens']
                event['final_budget'] = deepcopy(decision)
                event['count_scope'] = 'candidate_in_last_accepted_task_frame'

    def _budget_rejection(self, code, message=None):
        available = self._exact_limits().available_prompt_tokens if self._budget_identity else None
        return {'allowed': False, 'code': code, 'message': message or code,
                'budget_class': 'rejected', 'required_tokens': None,
                'available_tokens': available, 'budget_mode': self.config.budget_mode}

    def _pending_group_return_plan(self, request, route):
        """Prove one ordinary tool completion since the last accepted task send.

        This is a per-request classification, never a standing excursion. Both
        the canonical prefix and the counted wire prefix must be unchanged,
        except for the engine's explicit pressure-notice transition. Other host
        additions cannot borrow the tool reserve.
        """
        from .protocol import validate_tool_protocol

        frame = self._budget_frame
        if (frame is None or frame['conversation_id'] != self.store.conversation_id
                or frame['route'] != route):
            return None
        prior_uids = frame['source_uids']
        if (not prior_uids or len(self.store.order) <= len(prior_uids)
                or self.store.order[:len(prior_uids)] != prior_uids):
            return None
        tail = [self.store.messages[uid].envelope for uid in self.store.order[len(prior_uids):]]
        canonical = validate_tool_protocol(tail)
        if (not canonical.valid or len(canonical.groups) != 1
                or not canonical.groups[0].complete
                or canonical.groups[0].source_indices != tuple(range(len(tail)))):
            return None
        before = effective_request_body(frame['request'])
        current = effective_request_body(request)
        old_messages = before.pop('messages', None)
        messages = current.pop('messages', None)
        if (before != current or request.get('extra_headers') != frame['request'].get('extra_headers')
                or type(old_messages) is not list or type(messages) is not list):
            return None
        old_messages = self._prefix_with_current_pressure_notice(old_messages, messages, frame)
        if (old_messages is None or messages[:len(old_messages)] != old_messages
                or messages != self._render_budget_messages(self.store)):
            return None
        wire_tail = messages[len(old_messages):]
        new_echo = (self._exact_pressure_notice or {}).get('tail_echo') or ''
        if new_echo and wire_tail and wire_tail[-1] == {'role': 'system', 'content': new_echo}:
            # The engine-owned tail reminder is not part of the pending tool group.
            wire_tail = wire_tail[:-1]
        wire = validate_tool_protocol(wire_tail)
        if (not wire.valid or len(wire.groups) != 1 or not wire.groups[0].complete
                or wire.groups[0].source_indices != tuple(range(len(wire_tail)))
                or wire.groups[0].call_ids != canonical.groups[0].call_ids):
            return None
        return (f'Complete the pending tool group, then compact or archive stale context '
                f'below {self._exact_limits().normal_threshold} tokens before further expansion.')

    def _count_budget_request(self, request, route, *, reason=None, return_plan=None, allow_pending=False,
                              pending_group_plan=None, protected_group_plan=None):
        self._require_origin()
        self.config.validate_budget()
        if self._exact_counter is None or self._budget_identity is None:
            return self._budget_rejection('exact_counter_unconfigured')
        if self._budget_binding_invalid:
            return self._budget_rejection('provider_identity_changed')
        identity = self._budget_identity
        if type(route) is dict and route.get('unattested_transport_options'):
            return self._budget_rejection('unsupported_transport_options',
                'Exact counting requires no unpropagated client query/header overrides and zero SDK retries')
        if (type(route) is not dict or route.get('api_mode') != 'chat_completions'
                or route.get('provider') != identity.provider
                or str(route.get('base_url', '')).rstrip('/') != identity.base_url.rstrip('/')):
            return self._budget_rejection('provider_identity_changed')
        try:
            from .protocol import validate_tool_protocol
            messages = effective_request_body(request).get('messages')
            if type(messages) is not list or not messages:
                return self._budget_rejection('invalid_final_protocol', 'A nonempty final message list is required')
            protocol = validate_tool_protocol(messages, allow_pending=allow_pending)
            if not protocol.valid:
                refusal = self._budget_rejection('invalid_final_protocol',
                    'Final provider messages contain an orphaned, duplicate or incomplete tool group')
                refusal['protocol_errors'] = [deepcopy(vars(item)) for item in protocol.violations]
                return refusal
            count = self._exact_counter.count_final_request(request, identity)
            if count.identity != identity:
                return self._budget_rejection('provider_identity_changed')
            # Ordinary completion grants only internal reserve. A separately
            # proven protected completion may use bounded provider headroom.
            # Normal requests retain their ordinary classification.
            limits = self._exact_limits()
            if reason is None and pending_group_plan and limits.normal_threshold < count.tokens <= limits.active_target:
                reason, return_plan = 'pending_group', pending_group_plan
            elif reason is None and protected_group_plan and count.tokens > limits.active_target:
                reason, return_plan = 'protected_group', protected_group_plan
            decision = enforce_final_request(request, self._exact_limits(), count=count,
                                             reason=reason, return_plan=return_plan).to_dict()
        except CounterError as exc:
            return self._budget_rejection(exc.code, str(exc))
        except Exception as exc:
            return self._budget_rejection('exact_count_failed', f'{type(exc).__name__}: {exc}')
        decision['budget_mode'] = 'exact'
        return decision

    def check_final_request(self, request, *, route, purpose='task'):
        if self._pending_direct_rollback is not None:
            try:
                self.retry_context_rollback()
            except Exception as error:
                return self._budget_rejection('context_rollback_incomplete',
                    f'Direct context rollback remains incomplete: {type(error).__name__}: {error}')
            return self._budget_rejection('context_rollback_recovered',
                'The failed context edit was rolled back; rebuild the request from the accepted state')
        self._require_origin()
        self.config.validate_budget()
        if self.config.budget_mode == 'approximate':
            return {'allowed': True, 'code': 'approximate_legacy_mode',
                    'budget_class': 'approximate', 'budget_mode': 'approximate'}
        if self._pending_tool_budget is not None:
            pending = self._pending_tool_budget
            self.abort_tool_result(pending['name'], pending['call_id'])
            return self._budget_rejection('tool_admission_pending', 'A context tool was not admitted by the host before dispatch')
        excursion = self._budget_excursion or {}
        completion_plan = (self._pending_group_return_plan(request, route)
                           if purpose == 'task' and not excursion else None)
        protected_plan = None
        if completion_plan:
            # The prefix/route/schema/group proof above must succeed first.
            # Protection alone cannot classify unrelated growth, auxiliary
            # requests, repeated sends, or an incomplete result group.
            new_uids = self.store.order[len(self._budget_frame['source_uids']):]
            if new_uids and all(self.store.messages[uid].protected for uid in new_uids):
                protected_plan = completion_plan
        decision = self._count_budget_request(request, route,
            reason=excursion.get('reason'), return_plan=excursion.get('return_plan'),
            pending_group_plan=completion_plan, protected_group_plan=protected_plan)
        if purpose == 'task' and self.config.context_map_enabled:
            try:
                measurement = self._final_context_map_measurement(request, route, decision)
                if measurement is not None:
                    decision['context_map'] = measurement
            except PolicyViolation as error:
                decision = self._budget_rejection(error.code, str(error))
                decision.update(required_tokens=error.required_tokens, available_tokens=error.available_tokens)
        decision['purpose'] = purpose
        affects_pressure = purpose == 'task' or decision.get('budget_class') != 'normal'
        if decision['allowed'] and affects_pressure and not self._pressure_episode.admit(
                decision['prompt_tokens'], decision['reason'], decision['return_plan'], self._exact_limits()):
            decision.update(allowed=False, code='pressure_episode_exhausted',
                            budget_class='rejected', message='pressure_episode_exhausted')
        if decision['allowed']:
            if purpose == 'task':
                self._budget_frame = {'request': deepcopy(request), 'route': deepcopy(route),
                                      'decision': deepcopy(decision),
                                      'pressure_notice': self._pressure_notice_text(),
                                      'tail_echo': (self._exact_pressure_notice or {}).get('tail_echo') or None,
                                      'conversation_id': self.store.conversation_id,
                                      'source_uids': list(self.store.order),
                                      'context_map_view': deepcopy(self._map_view_for(self.store))}
            if decision['budget_class'] == 'normal' and purpose == 'task':
                self._budget_excursion = None
        self._last_budget_decision = deepcopy(decision)
        return decision

    def exact_budget_status(self):
        decision = self._last_budget_decision
        return {'budget_mode': self.config.budget_mode,
                'exact_counter_bound': self._exact_counter is not None,
                # This is evidence about the last request, never a permanent certificate.
                'exact_budget_qualified': bool(self.config.budget_mode == 'exact' and decision
                                               and decision.get('allowed')),
                'last_final_budget': deepcopy(decision),
                'pressure_attempts': self._pressure_episode.attempts,
                'pressure_nonprogress': self._pressure_episode.nonprogress}
