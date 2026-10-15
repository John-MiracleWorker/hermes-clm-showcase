"""Telemetry (spec §30, §31): event log + metrics counters."""

from __future__ import annotations

from dataclasses import asdict
from copy import deepcopy
from datetime import datetime, timezone
from functools import wraps
import json
import os
import time
from typing import Any, Dict, List
from zoneinfo import ZoneInfo

_EVENT_KEYS = {
    "compact": ("source_message_ids", "summary_id"),
    "archive": ("source_message_ids", "reason"),
    "restore": ("source_message_ids", "summary_id"),
    "replace_summary": ("summary_id", "revision"),
    "rejection": ("code", "message"),
    "overflow": ("active_tokens", "retries_left"),
    "emergency_compaction": ("source_message_ids", "tokens_before", "tokens_after"),
    "nudge": ("level", "active_tokens"),
    "toolobs_offload": ("ref", "chars"),
    "session_start": (),
    "session_end": (),
    "session_reset": (),
    "reset_working_set": (),
}


def restore_timestamps():
    """UTC and local display timestamps for the same restore observation."""
    now = datetime.now(timezone.utc)
    return dict(timestamp_utc=now.isoformat(),
                timestamp_detroit=now.astimezone(ZoneInfo('America/Detroit')).isoformat())


def observe_reference_restore(operation):
    """Observe the outer loader call without writing its rejected transaction.

    A native success is only staged here; prepared-reply admission owns its
    eventual outcome. Nested tool/engine calls share this service-local depth.
    """
    @wraps(operation)
    def observed(engine, *args, **kwargs):
        if operation.__name__ == 'handle_tool_call':
            name = args[0] if args else kwargs.get('name')
            arguments = args[1] if len(args) > 1 else kwargs.get('args')
            requested = name == 'clm_restore' and type(arguments) is dict and 'archive_refs' in arguments
        elif operation.__name__ == 'tool_restore':
            arguments = args[0] if args else kwargs.get('args')
            requested = type(arguments) is dict and 'archive_refs' in arguments
        else:
            requested = 'archive_refs' in kwargs
        if not requested:
            return operation(engine, *args, **kwargs)
        with engine._tool_budget_lock:
            pending_before = engine._pending_tool_budget
            telemetry = engine.telemetry
            observer = telemetry._reference_restores
            outer = observer['depth'] == 0
            observer['depth'] += 1
            try:
                result = operation(engine, *args, **kwargs)
            except BaseException as error:
                if outer:
                    pending = engine._pending_tool_budget
                    telemetry.observe_restore_outcome(dict(ok=False,
                        error=getattr(error, 'code', 'restore_failed')),
                        pending=pending if pending is not pending_before else None)
                raise
            finally:
                observer['depth'] -= 1
            if outer:
                payload = json.loads(result) if type(result) is str else result
                if not payload.get('ok') or engine._pending_tool_budget is None:
                    pending = engine._pending_tool_budget
                    telemetry.observe_restore_outcome(payload,
                        pending=pending if pending is not pending_before else None)
            return result
    return observed


class Telemetry:
    """Event log (events.jsonl) + metrics bookkeeping over store.metrics."""

    def __init__(self, store: Any, state_root: str = ""):
        self.store = store
        self.state_root = state_root
        self._reference_restores = dict(depth=0, recent=[], counters={key: 0 for key in (
            'restore_success', 'restore_failure', 'restore_calls_from_lookup',
            'invalid_archive_ref_attempts', 'restored_tokens', 'noop_restores')})

    def observe_restore_outcome(self, result, *, pending=None):
        """Bounded current-instance observations, never checkpoint publications."""
        candidate = {}
        if pending is not None and pending.get('name') == 'clm_restore':
            if pending.get('restore_observation_done'):
                return
            pending['restore_observation_done'] = True
            candidate = pending.get('restore_result', {})
        result = {**candidate, **result}
        ok = result.get('ok') is True
        lookup_ids = list(result.get('lookup_ids') or [])
        error = None if ok else result.get('error', 'restore_failed')
        counters = self._reference_restores['counters']
        tokens = result.get('tokens_restored', 0) if ok else 0
        contribution = {
            'restore_success' if ok else 'restore_failure': 1,
            'restore_calls_from_lookup': int(bool(lookup_ids)),
            'invalid_archive_ref_attempts': int(error in (
                'invalid_archive_refs', 'unknown_archive_ref', 'historic_branch_conflict')),
            'restored_tokens': tokens,
            'noop_restores': int(ok and result.get('noop') is True),
        }
        for key, value in contribution.items():
            counters[key] += value
        row = dict(ok=ok, error=error, lookup_ids=lookup_ids,
                   archive_refs=list(result.get('archive_refs') or []),
                   restored_tokens=tokens, noop=ok and result.get('noop') is True,
                   observed_monotonic_ns=time.monotonic_ns(), whole_call_latency_ms=None,
                   **restore_timestamps())
        self._reference_restores['recent'] = (self._reference_restores['recent'] + [row])[-32:]
        if pending is not None and pending.get('name') == 'clm_restore':
            pending['restore_observation'] = (row, contribution)

    def cancel_restore_publication(self, pending):
        """Correct only this admitted call, even after its recent row was evicted."""
        observation = pending.get('restore_observation')
        if observation is None or not observation[0]['ok']:
            return
        row, contribution = observation
        for key, value in contribution.items():
            self._reference_restores['counters'][key] -= value
        self._reference_restores['recent'] = [item for item in self._reference_restores['recent']
                                             if item is not row]
        pending['restore_observation_done'] = False
        self.observe_restore_outcome(dict(ok=False, error='restore_publication_failed'), pending=pending)

    def reference_restore_snapshot(self):
        return dict(scope='current_engine_instance', persistent=False,
            tokens_scope='source_delta_estimate', **self._reference_restores['counters'],
            recent=deepcopy(self._reference_restores['recent']),
            source_consultation=None, wrong_restore=None, autonomous_success=None)

    def record(self, kind: str, initiator: str = "model", **fields: Any) -> Dict[str, Any]:
        event: Dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event": kind,
            "context_revision_before": fields.get("revision_before", self.store.revision),
            "context_revision_after": fields.get("revision_after", self.store.revision),
            "initiator": initiator,
        }
        # Extra fields (tokens_*, reason, level, …) ride along.
        for key, value in fields.items():
            if key not in {"revision_before", "revision_after", "metrics"}:
                event[key] = value
        source_ids = set(event.get("source_message_ids") or [])
        if source_ids:
            event["tool_groups"] = [asdict(group) for group in self.store.groups()
                                    if source_ids.intersection(group.source_uids)]
        if self.store.config.event_log:
            self.store.append_event(self.state_root, event)
        self._update_metrics(kind, event)
        return event

    def _update_metrics(self, kind: str, event: Dict[str, Any]) -> None:
        m = self.store.metrics
        m["total_events"] = m.get("total_events", 0) + 1
        if kind == "compact":
            m["context_compactions"] = m.get("context_compactions", 0) + 1
            m["tokens_removed"] = m.get("tokens_removed", 0) + event.get("tokens_saved", 0)
            m["summary_tokens_created"] = m.get("summary_tokens_created", 0) + event.get("summary_tokens", 0)
        elif kind == "archive":
            m["context_archives"] = m.get("context_archives", 0) + 1
            m["tokens_removed"] = m.get("tokens_removed", 0) + event.get("tokens_removed", 0)
        elif kind == "restore":
            m["context_restores"] = m.get("context_restores", 0) + 1
            m["tokens_restored"] = m.get("tokens_restored", 0) + event.get("tokens_restored", 0)
        elif kind == "lookup":
            m["context_lookups"] = m.get("context_lookups", 0) + 1
            m["lookup_candidates_presented"] = m.get("lookup_candidates_presented", 0) + event['candidate_count']
        elif kind == "rejection":
            m["context_rejections"] = m.get("context_rejections", 0) + 1
        elif kind == "overflow":
            m["overflow_events"] = m.get("overflow_events", 0) + 1
        elif kind == "emergency_compaction":
            m["emergency_host_compactions"] = m.get("emergency_host_compactions", 0) + 1
        # Thrash tracking (spec §47.4): per-uid archive/restore oscillation.
        source_ids = (event.get('changed_source_ids', event.get('source_message_ids', []))
                      if kind == 'restore' else event.get('source_message_ids', []))
        for uid in source_ids or []:
            key = f"thrash::{uid}"
            if kind == "restore":
                m[key] = m.get(key, 0) + 1
            elif kind == "archive":
                m[key] = 0  # archive resets oscillation for this uid

    def model_call_metrics(self, prompt_tokens: int, completion_tokens: int) -> None:
        m = self.store.metrics
        m["total_model_calls"] = m.get("total_model_calls", 0) + 1
        m["total_prompt_tokens"] = m.get("total_prompt_tokens", 0) + prompt_tokens
        m["total_completion_tokens"] = m.get("total_completion_tokens", 0) + completion_tokens

    def snapshot(self) -> Dict[str, Any]:
        m = dict(self.store.metrics)
        for key in list(m.keys()):
            if key.startswith("thrash::"):
                del m[key]
        return m

    def recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        if not self.store.conversation_id:
            return []
        path = os.path.join(self.state_root, self.store.conversation_id, "events.jsonl")
        if not os.path.exists(path):
            return []
        events: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return events[-limit:]
