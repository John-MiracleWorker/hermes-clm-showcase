"""Exact context tools publish only after the host prepares their real reply."""
from copy import deepcopy
from contextlib import ExitStack
import json
from threading import RLock

from .policies import PolicyViolation
from .provider_budget import effective_request_body
from .protocol import envelope_digest, envelope_uid, provider_envelope, validate_json, validate_tool_protocol
from .tool_publication import ToolPublicationMixin


_MUTATING_TOOLS = frozenset(('clm_compact', 'clm_archive', 'clm_restore', 'clm_replace_summary', 'clm_lookup'))


class _ToolAdmissionLock:
    """Synchronization is an owned service, not copied runtime state.

    Detached runtime inspection retains the service identity. Engine clones
    construct their own lock; transaction rollback never replaces a held lock.
    """
    def __init__(self):
        self._lock = RLock()

    def __enter__(self):
        return self._lock.__enter__()

    def __exit__(self, *args):
        return self._lock.__exit__(*args)

    def __deepcopy__(self, memo):
        return self


def _refusal(error):
    result = {'ok': False, 'error': error.code, 'message': error.message}
    for key in ('required_tokens', 'available_tokens'):
        value = getattr(error, key, None)
        if value is not None:
            result[key] = value
    return json.dumps(result, ensure_ascii=False)


def _require_allowed(decision):
    if not decision['allowed']:
        raise PolicyViolation(decision['code'], decision['message'],
            required_tokens=decision.get('required_tokens'), available_tokens=decision.get('available_tokens'))


class ToolBudgetMixin(ToolPublicationMixin):
    def _initialize_tool_budget(self):
        self._pending_tool_budget = None
        self._pending_direct_rollback = None
        self._pending_tool_publication = None
        self._tool_budget_lock = _ToolAdmissionLock()
        self.store._transaction_admission = self._require_context_rollback_complete

    def _require_context_rollback_complete(self, *, inspection=False):
        if self._pending_direct_rollback is not None:
            raise PolicyViolation('context_rollback_incomplete',
                'A direct context edit must finish rollback before normal operations can continue')
        pending = self._pending_tool_budget
        if pending is not None and pending.get('rollback_failed'):
            raise PolicyViolation('tool_rollback_incomplete',
                'The owned context edit must finish rollback before normal operations can continue')
        publication = self._pending_tool_publication
        if publication is not None:
            if publication.get('rollback_failed'):
                raise PolicyViolation('tool_rollback_incomplete',
                    'The unpublished context edit must finish rollback before normal operations can continue')
            if not inspection:
                raise PolicyViolation('tool_publication_pending',
                    'The native host must resolve its context tool publication before further operations')

    def _restore_context_edit(self, snapshot, files, services=None):
        try:
            if services is not None:
                self.__dict__.update(services)
            self._rollback_transaction(snapshot)
        finally:
            if files is not None:
                self.store._restore_session_files(files)

    def _retain_direct_rollback(self, snapshot, files, services=None, shared_owners=()):
        self._pending_direct_rollback = {'snapshot': snapshot, 'files': files,
                                         'services': services, 'shared_owners': shared_owners}

    def _rollback_direct_edit(self, snapshot, files, services=None):
        if self._pending_tool_budget is not None:
            # The enclosing native tool retains the original cancellation owner.
            self._restore_context_edit(snapshot, files, services)
            return
        self._retain_direct_rollback(snapshot, files, services)
        self.retry_context_rollback()

    def retry_context_rollback(self):
        """Retry owned direct rollback without ordinary writes or callbacks.

        Failure retains the preimage; success permits normal operations again.
        Final-request admission is a concrete recovery consumer.
        """
        pending = self._pending_direct_rollback
        owners = pending.get('shared_owners') or (self,) if pending is not None else (self,)
        # Shared-checkpoint startup preimages remain gated together: clearing a
        # sibling early would let its next publication precede a stale retry.
        # These are existing in-process owners, not a durable store coordinator.
        with ExitStack() as locks:
            for owner in sorted(owners, key=id):
                locks.enter_context(owner._tool_budget_lock)
            pending = self._pending_direct_rollback
            if pending is None:
                return False
            failure = None
            for owner in owners:
                owned = owner._pending_direct_rollback
                if owned is None:
                    continue
                try:
                    owner._restore_context_edit(owned['snapshot'], owned['files'], owned.get('services'))
                except BaseException as error:
                    if failure is None:
                        failure = error
            if failure is not None:
                raise failure
            for owner in owners:
                owner._pending_direct_rollback = None
            return True

    def _publish_context_transaction(self):
        if self._pending_tool_budget is None:
            self.store.commit_transaction(self._state_root,
                require_event_publication=(self.config.budget_mode == 'exact'
                                           and self._budget_operation is not None)
                    or any('context_map_delta' in event or event.get('event') in ('context_map_backfill', 'lookup')
                           for event in self.store._event_buffer or ()))

    def _current_tool_tail(self, name, call_id, messages):
        validate_json(messages)
        size = len(self.store.order)
        if (type(messages) is not list or len(messages) <= size
                or [envelope_uid(i, m) for i, m in enumerate(messages[:size])] != self.store.order):
            raise PolicyViolation('candidate_tool_frame_mismatch', 'Current tool context does not extend the accepted canonical branch')
        for uid, message in zip(self.store.order, messages[:size]):
            if envelope_digest(message) != envelope_digest(self.store.messages[uid].envelope):
                raise PolicyViolation('source_envelope_conflict',
                    'Current tool context changed a retained complete canonical envelope')
        tail = deepcopy(messages[size:])
        matches = [c for m in tail if m.get('role') == 'assistant'
                   for c in m.get('tool_calls') or [] if c.get('id') == call_id]
        if (type(call_id) is not str or not call_id or len(matches) != 1
                or matches[0].get('function', {}).get('name') != name
                or any(m.get('role') == 'tool' and m.get('tool_call_id') == call_id for m in tail)
                or not validate_tool_protocol(messages, allow_pending=True).valid):
            raise PolicyViolation('candidate_tool_frame_mismatch', 'Current tool call identity or protocol cannot be proved')
        return tail

    def _tool_candidate_request(self, store, frame, tail, *, with_map_measurement=False):
        from .retrieval.snapshot import attach_view
        from .map_snapshot import replace_messages
        request = deepcopy(frame['request'])
        base = self._base_budget_messages(store)
        view = self._map_view_for(store, frame.get('context_map_view') if store is not self.store else None)
        messages = attach_view(base, view) + [provider_envelope(m) for m in tail]
        if self._request_renderer is not None:
            messages = self._request_renderer(messages)
        extra = request.get('extra_body')
        if type(extra) is dict and 'messages' in extra:
            extra['messages'] = messages
        else:
            request['messages'] = messages
        measurement = None
        if view is not None and self.config.budget_mode == 'exact':
            without = replace_messages(request, self._render_map_messages(base + [provider_envelope(m) for m in tail]))
            measurement = self._measure_context_map(request, without, frame['route'], allow_pending=True)
            self._require_map_cap(measurement)
        return (request, measurement) if with_map_measurement else request

    def _check_known_tool_tail(self, frame, reason, return_plan):
        pending = self._pending_tool_budget
        if pending is None:
            return
        request = self._tool_candidate_request(self.store, frame, pending['tail'])
        decision = self._count_budget_request(request, frame['route'], reason=reason,
            return_plan=return_plan, allow_pending=True)
        _require_allowed(decision)

    def _dispatch_budgeted_tool(self, name, args, messages, call_id):
        from .tools import dispatch
        with self._tool_budget_lock:
            self._require_origin()
            if self._pending_tool_budget is not None:
                raise PolicyViolation('tool_admission_pending', 'A previous context tool still awaits its host reply admission')
            self._require_candidate_frame(self.store)
            tail = self._current_tool_tail(name, call_id, messages)
            pending = {'name': name, 'call_id': call_id, 'tail': tail,
                       'snapshot': self._transaction_snapshot(), 'frame': deepcopy(self._budget_frame)}
            self._pending_tool_budget = pending
            try:
                reply = dispatch(self, name, args or {})
                result = json.loads(reply)
                if type(result) is not dict or not result.get('ok'):
                    self._rollback_tool_stage(pending)
                elif self.store._event_buffer is None:
                    raise PolicyViolation('tool_admission_missing_transaction', 'Context edit has no staged transaction')
                elif name == 'clm_lookup':
                    pending['lookup_result'] = result
                elif name == 'clm_restore' and 'archive_refs' in args:
                    pending['restore_result'] = result
                return reply
            except BaseException:
                self._rollback_tool_stage(pending)
                raise

    def _rollback_tool_stage(self, pending):
        """Restore the owned stage, including any incomplete file publication."""
        pending['publishing'] = False
        pending['rollback_failed'] = True
        self._restore_context_edit(pending['snapshot'], pending.get('publication_files'))
        # A failed rollback retains its owner so outer cleanup can retry and
        # the final request guard cannot treat the partial stage as admitted.
        self._pending_tool_budget = None

    def abort_tool_result(self, name, call_id):
        """Restore only our own uncommitted stage; no save or lifecycle callback."""
        with self._tool_budget_lock:
            pending = self._pending_tool_budget
            if pending is not None and (name, call_id) == (pending['name'], pending['call_id']):
                try:
                    self._rollback_tool_stage(pending)
                finally:
                    self._observe_tool_restore(pending, dict(ok=False, error='restore_cancelled'))
                return True
            return False

    def _observe_tool_restore(self, pending, result):
        if 'restore_result' in pending:
            self.telemetry.observe_restore_outcome(result, pending=pending)

    def admit_tool_result(self, name, call_id, messages, tool_message, *, is_error=False, blocked=False):
        """Called with the host's prepared reply before canonical append.

        Only known current envelopes are counted. Unfinished sibling calls are
        explicitly pending; final dispatch still requires a complete group.
        None means admitted/no staged edit; a JSON string replaces a refused reply.
        """
        with self._tool_budget_lock:
            pending = self._pending_tool_budget
            if pending is None or (name, call_id) != (pending['name'], pending['call_id']):
                return None
            try:
                self._require_origin()
                if is_error or blocked:
                    try:
                        self._rollback_tool_stage(pending)
                    finally:
                        self._observe_tool_restore(pending, dict(ok=False,
                            error='restore_blocked' if blocked else 'restore_error_reply'))
                    return None
                tail = self._current_tool_tail(name, call_id, messages)
                validate_json(tool_message)
                if (type(tool_message) is not dict or tool_message.get('role') != 'tool'
                        or tool_message.get('tool_call_id') != call_id):
                    raise PolicyViolation('candidate_tool_frame_mismatch', 'Host prepared a different tool result')
                if name == 'clm_lookup' or 'restore_result' in pending:
                    result_key = 'lookup_result' if name == 'clm_lookup' else 'restore_result'
                    try:
                        content = tool_message.get('content')
                        # JSON booleans and numbers must stay distinct (True == 1 in Python).
                        unchanged = type(content) is str and json.dumps(json.loads(content), sort_keys=True) == json.dumps(pending[result_key], sort_keys=True)
                    except (ValueError, TypeError):
                        unchanged = False
                    if not unchanged:
                        raise PolicyViolation(result_key + '_transformed',
                            'Prepared tool result differs from the owned candidate result')
                tail.append(deepcopy(tool_message))
                from .store import ContextStore
                before_store = ContextStore(self.config, self.counter)
                before_store._restore_snapshot(pending['snapshot']['store'])
                frame = pending['frame']
                excursion = self._budget_excursion or {}
                reason = 'restore' if name == 'clm_restore' else excursion.get('reason')
                options = dict(reason=reason, return_plan=excursion.get('return_plan'), allow_pending=True)
                before_request, before_map = self._tool_candidate_request(before_store, frame, tail, with_map_measurement=True)
                after_request, after_map = self._tool_candidate_request(self.store, frame, tail, with_map_measurement=True)
                before = self._count_budget_request(before_request, frame['route'], **options)
                after = self._count_budget_request(after_request, frame['route'], **options)
                if before_map is not None:
                    before['context_map'] = before_map
                if after_map is not None:
                    after['context_map'] = after_map
                # A too-large prior view can be repaired; an unavailable count cannot.
                if before.get('prompt_tokens') is None:
                    _require_allowed(before)
                    raise PolicyViolation('exact_count_failed', 'No exact count for the prior candidate')
                _require_allowed(after)
                limits = self._exact_limits()
                if (after['prompt_tokens'] > limits.normal_threshold
                        and (self._pressure_episode.nonprogress >= self._pressure_episode.max_nonprogress
                             or self._pressure_episode.attempts >= self._pressure_episode.max_attempts)):
                    raise PolicyViolation('pressure_episode_exhausted', 'pressure_episode_exhausted',
                        required_tokens=after['prompt_tokens'], available_tokens=limits.normal_threshold)
                growth = max(0, after['prompt_tokens'] - before['prompt_tokens'])
                cap = self.config.max_restoration_tokens
                if reason == 'restore' and cap and growth > cap:
                    raise PolicyViolation('max_restoration_exceeded', 'Rendered restoration exceeds the configured limit',
                        required_tokens=growth, available_tokens=cap)
                self._pressure_episode.observe_edit(before['prompt_tokens'], after['prompt_tokens'], limits.normal_threshold)
                if after['budget_class'] == 'normal':
                    self._budget_excursion = None
                elif reason == 'restore':
                    self._budget_excursion = {'reason': reason, 'return_plan': excursion.get('return_plan')}
                after['count_scope'] = 'candidate_known_tool_frame'
                after['tool_group_complete'] = validate_tool_protocol(effective_request_body(after_request)['messages']).valid
                self._last_budget_decision = deepcopy(after)
                for event in self.store._event_buffer or ():
                    if event.get('event') in ('compact', 'compact_body', 'archive', 'restore', 'replace_summary', 'lookup'):
                        event.update(exact_prompt_before=before['prompt_tokens'], exact_prompt_after=after['prompt_tokens'],
                                     final_budget=deepcopy(after), count_scope=after['count_scope'])
                if self._conversation_id and (self.config.persistence or self.config.event_log):
                    pending['publication_files'] = self.store._snapshot_session_files(
                        self._state_root, self._conversation_id)
                # Keep cancellation ownership until every publication completes.
                # Only this phase may persist a pending tool transaction.
                pending['publishing'] = True
                self._persist()
                self.store.commit_transaction(self._state_root, require_event_publication=True)
                self._pending_tool_budget = None
                self._observe_tool_restore(pending, dict(ok=True))
                if pending.get('host_publication'):
                    self._pending_tool_publication = pending
                return None
            except BaseException as error:
                try:
                    self._rollback_tool_stage(pending)
                finally:
                    self._observe_tool_restore(pending, dict(ok=False,
                        error=getattr(error, 'code', 'tool_admission_failed')))
                if not isinstance(error, Exception):
                    raise
                if self._origin_mode == 'native':
                    from agent.context_engine import SourceProjectionError
                    if isinstance(error, SourceProjectionError):
                        raise
                if isinstance(error, PolicyViolation):
                    return _refusal(error)
                return json.dumps({'ok': False, 'error': 'tool_admission_failed', 'message': str(error)})
