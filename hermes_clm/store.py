"""Reversible canonical-envelope store with versioned identities.

Canonical order includes every instruction. Historic versions remain archived
outside that order after rewinds. Text is only an indexing/budget aid.
"""

from __future__ import annotations

from copy import deepcopy
from functools import wraps
import hashlib
import json
import os
import tempfile
from typing import Any, Dict, List, Optional

from .models import (ContextMessage, ContextSummary, ProtectionSnapshot, utc_now_iso, _object,
                     _strings, _integers, _string_list, validate_metrics,
                     validate_body_compaction)
from .tokenizer import TokenCounter
from .protocol import (canonical_envelope, envelope_uid, tool_groups,
                       validate_tool_protocol, RecoveryRequired, ToolGroup)
from . import invariants
from .prompts import render_summary_message
from .protocol import (provider_envelope, envelope_digest, validate_coordinate_origin,
                       projection_refuse, validate_source_conflicts, validate_native_protocol)

_STATE_VERSION = 4
_NATIVE_FIELDS = frozenset({
    "version", "uid_version", "uid_migration", "legacy_state", "recovery_required",
    "conversation_id", "session_id", "system_message", "system_uid", "messages",
    "order", "summaries", "toolobs", "revision", "summary_seq", "toolobs_seq",
    "metrics", "budget_tokens", "reserve_tokens", "current_tokens",
})
_NATIVE_MESSAGE_FIELDS = frozenset({
    "uid", "role", "content", "source", "created_at", "token_count", "protected",
    "active", "archived", "summary_parent_ids", "metadata", "index", "toolobs_ref",
    "toolobs_chars", "envelope", "body_compaction", "body_compaction_history",
    "permanent_reasons", "temporary_reasons",
})
_NATIVE_SUMMARY_FIELDS = frozenset({
    "uid", "content", "source_message_ids", "created_by", "created_at", "token_count",
    "revision", "anchor_index", "active", "version", "derivation_source_uids",
})
_NATIVE_BODY_FIELDS = frozenset({
    "version", "content", "derivation_source_uids", "restore_group_uids",
    "created_by", "created_at",
})


def _accepted_mutation(method):
    """Independently callable mutations accept or roll back a complete state."""
    @wraps(method)
    def apply(self, *args, **kwargs):
        self.require_origin()
        self.validate_accepted()
        snapshot = self._snapshot()
        rollback_budget = None
        try:
            self.recompute_protection()
            if not self._budget_compound_depth:
                rollback_budget = self._check_exact_budget(method.__name__, 'before')
            result = method(self, *args, **kwargs)
            self._sync_context_map()
            self.validate_accepted()
            if self._projection_context is not None:
                self._validate_native_presentation()
            if not self._budget_compound_depth:
                self._check_exact_budget(method.__name__, 'after')
            return result
        except BaseException:
            try:
                if rollback_budget is not None:
                    rollback_budget(snapshot)
            finally:
                self._restore_snapshot(snapshot)
            raise
    return apply


def _content_of(message: Dict[str, Any]) -> str:
    """Extract indexing text only; never used as the v2 identity or wire body."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    return "" if content is None else str(content)


def _tool_call_fingerprint(message: Dict[str, Any]) -> str:
    tool_calls = message.get("tool_calls")
    if not tool_calls:
        return ""
    try:
        return json.dumps(tool_calls, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(tool_calls)


def message_uid(index: int, role: str, content: str, tool_calls_fp: str = "") -> str:
    """Legacy v1 identity helper retained for callers; v2 uses envelope_uid."""
    h = hashlib.sha256()
    h.update(f"{index}|{role}|{content}|{tool_calls_fp}".encode("utf-8", "replace"))
    return "msg_" + h.hexdigest()[:12]


def summary_uid(n: int) -> str:
    return f"clm_summary_{n:04d}"


def toolobs_uid(n: int) -> str:
    return f"toolobs_{n:05d}"


class ContextStore:
    """In-memory canonical state for one conversation (spec §39)."""

    def __init__(self, config: Any, counter: Optional[TokenCounter] = None):
        config.validate_retention()
        config.validate_context_map()
        self.config = config
        self.counter = counter or TokenCounter(getattr(config, "tokenizer", "auto"))
        self.messages: Dict[str, ContextMessage] = {}
        self.order: List[str] = []          # canonical order (uids, by index)
        self._by_index: Dict[int, str] = {}
        self.summaries: Dict[str, ContextSummary] = {}
        self.toolobs: Dict[str, Dict[str, Any]] = {}
        self.revision: int = 0
        self.conversation_id: str = ""
        self.session_id: str = ""
        self.metrics: Dict[str, Any] = {}
        self._summary_seq = 0
        self._toolobs_seq = 0
        self._dirty = False
        self.uid_migration: Dict[str, str] = {}
        self.legacy_state: Optional[Dict[str, Any]] = None
        self.recovery_required: Optional[str] = None
        self._event_buffer: Optional[List[dict]] = None
        self._event_publication_required = False
        self.coordinate_origin = None
        self._origin_mode = "standalone"
        self._origin_admission = "admitted"
        self._origin_reason = None
        self._canonical_ingestion = False
        self._projection_policy = {}
        self._projection_context = None
        self._budget_validator = None
        self._transaction_admission = None
        self._budget_compound_depth = 0
        self._context_map = None
        self._pending_context_map_backfill = None
        self._pending_retrofit_retirement = None
        self._pending_save_rollback = None

    @property
    def context_map_index(self):
        from .retrieval.context_map import ContextMapIndex
        return ContextMapIndex(self)

    def _sync_context_map(self, *, diagnostic=False):
        if self.config.context_map_enabled or self._context_map is not None:
            from .retrieval.context_map import sync_map
            sync_map(self, diagnostic=diagnostic)

    def _check_exact_budget(self, operation, phase):
        if self.config.budget_mode != 'exact':
            return
        if not callable(self._budget_validator):
            from .policies import PolicyViolation
            raise PolicyViolation('exact_counter_unconfigured', 'Exact candidate validator is not bound')
        return self._budget_validator(self, operation, phase)


    # ------------------------------------------------------------------ #
    # ingestion                                                           #
    # ------------------------------------------------------------------ #

    @property
    def system_message(self) -> Optional[Dict[str, Any]]:
        """Compatibility accessor for the first system instruction (detached)."""
        return next((self.messages[u].envelope for u in self.order
                     if self.messages[u].role == "system"), None)

    @property
    def system_uid(self) -> str:
        return next((u for u in self.order if self.messages[u].role == "system"), "")

    def require_origin(self, *, ingest=False, inspection=False) -> None:
        if self._origin_admission != "admitted":
            reason = self._origin_reason or {"code": "coordinate_resume_unproved", "path": "",
                                            "message": "Canonical coordinate origin is not admitted."}
            projection_refuse(**reason)
        if self._pending_save_rollback is not None:
            from .policies import PolicyViolation
            raise PolicyViolation('context_rollback_incomplete',
                                  'Standalone save rollback must finish before further operations')
        if self._transaction_admission is not None:
            if inspection:
                self._transaction_admission(inspection=True)
            else:
                self._transaction_admission()
        if ingest and self.coordinate_origin is not None and not self._canonical_ingestion:
            projection_refuse("projection_missing", "", "Marked history requires canonical host ingestion.")

    def require_envelopes(self) -> None:
        if self.recovery_required:
            raise RecoveryRequired(self.recovery_required)

    def validate_accepted(self, *, allow_pending=True, inspection=False) -> None:
        self.require_origin(inspection=inspection)
        self.require_envelopes()
        self.config.validate_context_map()
        self._validate_state_schema(self.state_dict())
        invariants.validate(invariants.capture(self), allow_pending=allow_pending)
        self._validate_protection_state()
        if self._context_map is not None:
            from .retrieval.context_map import validate_map
            validate_map(self, self._context_map)

    def _validate_protection_state(self) -> None:
        """Reject contradictory saved reasons before deriving a new window."""
        current = set(self.order)
        for uid, rec in self.messages.items():
            if (rec.permanent_reasons or rec.temporary_reasons) and not rec.protected:
                raise ValueError("inconsistent_protection_reasons")
            if uid in current and rec.protected and (not rec.active or rec.body_compaction is not None):
                raise ValueError("invalid_protected_presentation")

    def ingest_messages(self, messages: List[Dict[str, Any]],
                        created_at: Optional[str] = None) -> int:
        self.require_origin(ingest=True)
        self.require_envelopes()
        self.config.validate_context_map()
        if self.coordinate_origin is not None:
            validate_source_conflicts(messages, self.messages)
        snapshot = self._snapshot()
        try:
            result = self._ingest_messages(messages, created_at)
            self._projection_policy = {}
            self._projection_context = None
            return result
        except BaseException:
            self._restore_snapshot(snapshot)
            raise

    def _ingest_messages(self, messages: List[Dict[str, Any]],
                         created_at: Optional[str] = None, *, rolling_protection=True) -> int:
        self.require_envelopes()
        # Validate shape before mutation; pending and invalid associations can be
        # represented for diagnosis, but selection/operations must reject them.
        envelopes = [canonical_envelope(m) for m in messages]
        for envelope in envelopes:
            if not isinstance(envelope.get("role"), str) or not envelope["role"]:
                raise ValueError("canonical message requires its exact role")
        uids = [envelope_uid(i, m) for i, m in enumerate(envelopes)]
        previous = invariants.capture(self)
        invariants.validate(previous, allow_pending=True, diagnostic=True)
        self._validate_protection_state()
        if self.config.context_map_enabled or self._context_map is not None:
            from .retrieval.context_map import capture_witnesses
            capture_witnesses(self, previous)
        # Build the final branch before reconciling. Temporary tail removal is
        # not evidence that an unchanged sibling's archive choice was undone.
        for uid in set(self.order) - set(uids):
            self.messages[uid].active, self.messages[uid].archived = False, True
            self._dirty = True
        self.order = []
        self._by_index = {}
        added = 0
        stamp = created_at or utc_now_iso()
        for index, (envelope, uid) in enumerate(zip(envelopes, uids)):
            if previous.by_index.get(index) == uid:
                self.order.append(uid)
                self._by_index[index] = uid
                continue
            if uid in self.messages:
                record = self.messages[uid]
                record.active, record.archived = True, False
            else:
                record = ContextMessage(
                    uid=uid, role=envelope["role"], content=_content_of(envelope),
                    source="canonical", created_at=stamp,
                    token_count=self.counter.count_message(envelope), index=index,
                    metadata={"tool_call_ids": [c.get("id") for c in envelope.get("tool_calls") or []
                                              if isinstance(c, dict)],
                              "clm_pin": bool(envelope.get("metadata", {}).get("clm_pin"))
                              if isinstance(envelope.get("metadata"), dict) else False})
                record.set_envelope(envelope)
                self.messages[uid] = record
                added += 1
            self.order.append(uid)
            self._by_index[index] = uid
            self._dirty = True
        self.order = uids
        self._by_index = dict(enumerate(uids))
        plan = invariants.reconcile_canonical(previous, invariants.capture(self))
        for uid, flags in plan.residency.items():
            self.messages[uid].active, self.messages[uid].archived = flags
        for uid in plan.retire_bodies:
            self._retire_body_compaction(self.messages[uid])
        for uid in plan.retire_summaries:
            self.summaries[uid].active = False
        # Mandatory protection follows the final branch, including reused UIDs.
        first_user_uid = next((u for u in self.order if self.messages[u].role == "user"), None)
        for uid in self.order:
            self._apply_protection(self.messages[uid], first_user_uid)
        if rolling_protection:
            self._recompute_protection()
        # Direct shape-valid diagnostic ingestion can retain invalid protocol.
        # Healthy authoritative history recovers without validating that prior
        # diagnostic association as an accepted predecessor.
        invariants.validate(invariants.capture(self), allow_pending=True, diagnostic=True)
        self._sync_context_map(diagnostic=True)
        return added

    def _retire_body_compaction(self, rec: ContextMessage) -> None:
        if rec.body_compaction is not None:
            rec.body_compaction_history.append(deepcopy(rec.body_compaction))
            rec.body_compaction = None
            rec.token_count = self.counter.count_message(rec.envelope)
            self._dirty = True

    def _truncate_to_index(self, index: int) -> None:
        self.ingest_messages([self.messages[u].envelope for u in self.order
                              if self.messages[u].index < index])

    def _group_messages(self) -> List[Dict[str, Any]]:
        return [dict(self.messages[u].envelope, _clm_uid=u) for u in self.order]

    def groups(self) -> List[ToolGroup]:
        self.require_envelopes()
        return tool_groups(self._group_messages())

    def _source_groups(self, sources: List[str]) -> List[ToolGroup]:
        """Validate an ordered exact source branch, including archived versions.

        Current topology cannot validate an inactive historical summary: an
        unchanged result may now belong to a different assistant envelope.
        The retained source envelopes themselves must prove group closure.
        """
        return invariants.source_groups(invariants.capture(self), sources)

    def resolve_groups(self, uids: List[str], *, editing=True) -> List[str]:
        """Expand partial requests to group closure, or reject with stable codes."""
        from .policies import PolicyViolation
        self.require_envelopes()
        unknown = [u for u in uids if u not in self.messages]
        if unknown:
            raise PolicyViolation("unknown_message_ids", "unknown message ids", unknown)
        stale = [u for u in uids if u not in self.order]
        if stale:
            raise PolicyViolation("historic_branch_conflict",
                                  "historic versions require explicit canonical branch replay", stale)
        messages = self._group_messages()
        validation = (validate_native_protocol(messages) if self.coordinate_origin is not None
                      else validate_tool_protocol(messages, allow_pending=True))
        if not validation.valid:
            raise PolicyViolation(validation.violations[0].code, "invalid canonical tool protocol", uids)
        expanded = set(uids)
        for group in validation.groups:
            if expanded.intersection(group.source_uids):
                if editing and not group.complete:
                    raise PolicyViolation("pending_tool_group", "pending groups are immutable", list(group.source_uids))
                expanded.update(group.source_uids)
        return [u for u in self.order if u in expanded]

    def require_edit(self, uids: List[str], kind: str) -> None:
        """Only today's accepted host projection grants native edit permission."""
        self.require_origin()
        if self.coordinate_origin is None:
            return
        for uid in uids:
            rec = self.messages.get(uid)
            policy = self._projection_policy.get(uid)
            needs_permission = kind != "body" or (rec is not None and rec.role == "tool")
            if (uid not in self.order or rec is None or policy is None
                    or policy.get("essential") is not False
                    or (needs_permission and policy.get(kind) is not True)):
                projection_refuse("projection_ineligible", "", "Current host view does not permit this edit.")
            if kind == "body" and rec.role == "tool" and (type(rec.envelope.get("content")) is not str
                                   or not rec.envelope.get("content")):
                projection_refuse("projection_ineligible", "", "Structured or empty tool bodies cannot be edited.")
        # Permission does not relax complete current membership or pending refusal.
        self.resolve_groups(uids)

    def narrative_memberships(self):
        """Detached current presentations; the host supplies all origin authority."""
        return [{"members": list(s.source_message_ids), "derivation": list(s.derivation_source_uids)}
                for s in self.active_summaries() if self.is_summary_fully_archived(s)]

    def _validate_native_presentation(self, *, archive_uids=()):
        if self.coordinate_origin is None:
            return
        context = self._projection_context
        if context is None:
            projection_refuse("projection_missing", "", "A current accepted projection is required before narrative creation.")
        canonical, wire = context["canonical"], context["wire"]
        if (self.order != [envelope_uid(i, m) for i, m in enumerate(canonical)]
                or any(envelope_digest(self.messages[u].envelope) != envelope_digest(m)
                       for u, m in zip(self.order, canonical))):
            projection_refuse("projection_stale", "", "Placement authority belongs to a different canonical branch.")
        from agent.turn_context import _reissue_narrative_projection
        from .protocol import validate_source_projection
        from .selector import ContextSelector
        candidate = self
        if archive_uids:
            candidate = ContextStore(self.config, self.counter)
            candidate._restore_snapshot(self._snapshot())
            for uid in archive_uids:
                candidate.messages[uid].active, candidate.messages[uid].archived = False, True
            candidate._sync_context_map()
        candidate.validate_accepted()
        projection = _reissue_narrative_projection(
            context["projection"], canonical, wire, context["narratives"], candidate.narrative_memberships())
        validate_source_projection(projection, canonical, wire, context["incoming"], candidate.messages)
        return ContextSelector(candidate, None).build_projected(wire, projection)

    def _apply_protection(self, record: ContextMessage, first_user_uid: Optional[str]) -> None:
        reasons = []
        if record.role in {"system", "developer"}:
            reasons.append("required_instruction")
        if record.uid == first_user_uid:
            reasons.append("root_task")
        if record.metadata.get("clm_pin"):
            reasons.append("pinned")
        if reasons:
            record.protected = True
            record.active, record.archived = True, False
        for reason in reasons:
            if reason not in record.permanent_reasons:
                record.permanent_reasons.append(reason)
            if reason != "required_instruction":
                record.metadata.setdefault("protect_reason", reason)

    def protect_recent_turns(self, user_turns: int, assistant_turns: int) -> None:
        """Compatibility entry point; recent protection is derived, never sticky."""
        self.recompute_protection(recent_users=user_turns, recent_assistants=assistant_turns)

    def recompute_protection(self, *, recent_users=None, recent_assistants=None,
                             pending_group_ids=None) -> ProtectionSnapshot:
        """Recompute reasons from current order, preserving intrinsic protection.

        Completed tool-calling emissions count once. Closure reasons on sibling
        records are derived anew, so a changed tool association cannot inherit a
        historical sibling's permanent pin. Retained ghosts keep intrinsic
        reasons but never current-window or pending-execution reasons.
        """
        self.validate_accepted()
        snapshot = self._snapshot()
        try:
            result = self._recompute_protection(recent_users=recent_users,
                recent_assistants=recent_assistants, pending_group_ids=pending_group_ids)
            self._sync_context_map()
            self.validate_accepted()
            return result
        except BaseException:
            self._restore_snapshot(snapshot)
            raise

    def _recompute_protection(self, *, recent_users=None, recent_assistants=None,
                              pending_group_ids=None) -> ProtectionSnapshot:
        # Canonical ingestion also calls this private derivation on a staged
        # branch, including diagnostic pending/invalid protocol views. Public
        # callers must pass accepted-state validation before reaching it.
        self.require_origin()
        self.require_envelopes()
        self.config.validate_retention()
        users_n = self.config.protect_latest_user_turns if recent_users is None else recent_users
        assistants_n = self.config.protect_latest_assistant_turns if recent_assistants is None else recent_assistants
        if any(type(n) is not int or n < 0 for n in (users_n, assistants_n)):
            raise ValueError("recent retention must be nonnegative integers")
        groups = self.groups()
        pending = {g.assistant_uid for g in groups if not g.complete}
        if pending_group_ids is not None:
            if (not isinstance(pending_group_ids, set)
                    or not pending_group_ids.issubset({g.assistant_uid for g in groups})):
                raise ValueError("pending_group_ids must identify current tool groups")
            pending.update(pending_group_ids)
        snapshot = self._snapshot()
        try:
            temporary_names = {"recent_user", "recent_assistant", "pending_execution"}
            current = set(self.order)
            first_user = next((u for u in self.order if self.messages[u].role == "user"), None)
            for uid, rec in self.messages.items():
                prior_reasons = bool(rec.permanent_reasons or rec.temporary_reasons)
                permanent = {r for r in rec.permanent_reasons
                             if not r.startswith("tool_group:") and r not in temporary_names}
                legacy = rec.metadata.get("protect_reason")
                if legacy in {"root_task", "pinned"}:
                    permanent.add(legacy)
                if rec.protected and not prior_reasons and legacy not in temporary_names and not permanent:
                    permanent.add("explicit_protection")
                if uid in current:
                    if rec.role in {"system", "developer"}:
                        permanent.add("required_instruction")
                    if uid == first_user:
                        permanent.add("root_task")
                    if rec.metadata.get("clm_pin"):
                        permanent.add("pinned")
                rec.permanent_reasons = sorted(permanent)
                rec.temporary_reasons = []
                if legacy in temporary_names:
                    rec.metadata.pop("protect_reason", None)
            users = [u for u in self.order if self.messages[u].role == "user"]
            completed = [u for u in self.order if self.messages[u].role == "assistant" and u not in pending]
            for uid in users[-users_n:] if users_n else []:
                self.messages[uid].temporary_reasons.append("recent_user")
            for uid in completed[-assistants_n:] if assistants_n else []:
                self.messages[uid].temporary_reasons.append("recent_assistant")
            for group in groups:
                records = [self.messages[u] for u in group.source_uids]
                permanent = {r for rec in records for r in rec.permanent_reasons}
                temporary = {r for rec in records for r in rec.temporary_reasons}
                if group.assistant_uid in pending:
                    temporary.add("pending_execution")
                for rec in records:
                    rec.permanent_reasons = sorted(set(rec.permanent_reasons) |
                        {"tool_group:" + r for r in permanent if r not in rec.permanent_reasons})
                    rec.temporary_reasons = sorted(temporary)
            protected = set()
            for uid, rec in self.messages.items():
                rec.protected = bool(rec.permanent_reasons or rec.temporary_reasons)
                if uid in current and rec.protected:
                    protected.add(uid)
                    rec.active, rec.archived = True, False
                    self._retire_body_compaction(rec)
            # A newly protected source is shown raw, so a previous narrative
            # claiming that source no longer represents the current presentation.
            for summary in self.summaries.values():
                if summary.active and protected.intersection(summary.source_message_ids):
                    summary.active = False
            if self._snapshot() != snapshot:
                self._dirty = True
            return ProtectionSnapshot(
                tuple(u for u in self.order if u in protected),
                {u: tuple(self.messages[u].permanent_reasons) for u in self.order},
                {u: tuple(self.messages[u].temporary_reasons) for u in self.order})
        except Exception:
            self._restore_snapshot(snapshot)
            raise

    # ------------------------------------------------------------------ #
    # lookups                                                             #
    # ------------------------------------------------------------------ #

    def get(self, uid: str) -> Optional[ContextMessage]:
        return self.messages.get(uid)

    def summary(self, uid: str) -> Optional[ContextSummary]:
        return self.summaries.get(uid)

    def summary_by_prefix(self, uid: str) -> Optional[ContextSummary]:
        """Resolve shorthands: ``clm_summary_0009``, ``summary_0009``, ``0009``, ``9``."""
        if uid in self.summaries:
            return self.summaries[uid]

        def norm(value: str) -> str:
            value = value.replace("clm_summary_", "").replace("summary_", "").strip()
            return value.lstrip("0") or "0"

        target = norm(uid)
        for key, value in self.summaries.items():
            if norm(key) == target:
                return value
        return None

    def active_messages(self) -> List[ContextMessage]:
        return [self.messages[u] for u in self.order
                if self.messages[u].active and self.messages[u].role != "system"]

    def archived_messages(self) -> List[ContextMessage]:
        return [r for r in self.messages.values() if r.archived]

    def active_summaries(self) -> List[ContextSummary]:
        out = [s for s in self.summaries.values() if s.active]
        out.sort(key=lambda s: s.anchor_index)
        return out

    def is_summary_fully_archived(self, s: ContextSummary) -> bool:
        """A summary renders in context only when ALL its sources are archived
        (otherwise the raw message is already present)."""
        return all(
            (rec := self.messages.get(uid)) is not None and uid in self.order and rec.archived
            for uid in s.source_message_ids
        )

    # ------------------------------------------------------------------ #
    # mutations (atomic transaction: spec §33)                            #
    # ------------------------------------------------------------------ #

    def _snapshot(self) -> Dict[str, Any]:
        # Include flags, original bodies, revisions, counters and metrics, not
        # only the list of IDs. The counter/config are services, not state.
        return deepcopy({k: v for k, v in self.__dict__.items() if k not in {"config", "counter", "_budget_validator", "_transaction_admission", "_pending_save_rollback"}})

    @staticmethod
    def _validate_conversation_identity(conversation_id: str) -> None:
        """Require one plain UTF-8 filename component before owning files."""
        if (type(conversation_id) is not str or not conversation_id
                or conversation_id in {".", ".."} or "\x00" in conversation_id
                or os.path.basename(conversation_id) != conversation_id):
            raise ValueError("unsafe conversation identity")
        try:
            conversation_id.encode("utf-8", errors="strict")
        except UnicodeEncodeError:
            raise ValueError("unsafe conversation identity") from None

    @staticmethod
    def _snapshot_session_files(state_root: str, conversation_id: str) -> dict:
        """Capture only files owned by a native session lifecycle transaction."""
        ContextStore._validate_conversation_identity(conversation_id)
        root = os.path.abspath(state_root)
        base = os.path.join(root, conversation_id)
        files = {}
        for name in ("state.json", "events.jsonl"):
            path = os.path.join(base, name)
            if os.path.islink(path):
                raise OSError("A session transaction cannot own a symlink checkpoint")
            if os.path.exists(path):
                with open(path, "rb") as source:
                    files[path] = (source.read(), os.stat(path).st_mode & 0o7777)
            else:
                files[path] = None
        return {"files": files, "new_directories": [p for p in (base, root) if not os.path.exists(p)]}

    @staticmethod
    def _restore_session_files(snapshot: dict) -> None:
        """Undo publications without touching unrelated files in either session."""
        for path, saved in snapshot["files"].items():
            if saved is None:
                if os.path.exists(path):
                    os.unlink(path)
                continue
            data, mode = saved
            if os.path.exists(path):
                with open(path, "rb") as current:
                    # A byte-identical publication may still replace the mode.
                    if (current.read() == data
                            and os.fstat(current.fileno()).st_mode & 0o7777 == mode):
                        continue
            fd, temporary = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".state-rollback-")
            try:
                with os.fdopen(fd, "wb") as output:
                    output.write(data)
                os.chmod(temporary, mode)
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        for directory in snapshot["new_directories"]:
            if os.path.isdir(directory) and not os.listdir(directory):
                os.rmdir(directory)

    def begin_transaction(self) -> Dict[str, Any]:
        self.require_origin()
        if self._event_buffer is not None:
            raise RuntimeError("nested CLM transaction")
        snapshot = self._snapshot()
        self._event_buffer = []
        return snapshot

    def commit_transaction(self, state_root: str, *, require_event_publication: bool = False) -> None:
        self.require_origin()
        if self.coordinate_origin is not None:
            self.validate_accepted()
        events = self._event_buffer or []
        self._event_buffer = None
        previous = self._event_publication_required
        self._event_publication_required = require_event_publication
        try:
            for event in events:
                self.append_event(state_root, event)
        finally:
            self._event_publication_required = previous

    def _restore_snapshot(self, snap: Dict[str, Any]) -> None:
        for key in list(self.__dict__):
            if key not in {"config", "counter", "_budget_validator", "_transaction_admission", "_pending_save_rollback"}:
                del self.__dict__[key]
        self.__dict__.update(deepcopy(snap))

    @_accepted_mutation
    def create_summary(self, content: str, source_message_ids: List[str],
                       created_by: str = "model", *,
                       derivation_source_uids: Optional[List[str]] = None) -> ContextSummary:
        source_message_ids = self.resolve_groups(source_message_ids)
        self.require_edit(source_message_ids, "narrative")
        captured = invariants.capture(self)
        mandatory = invariants.derive_inputs(captured, source_message_ids)
        if derivation_source_uids is None:
            derivation = mandatory
        else:
            if type(derivation_source_uids) is not list:
                raise invariants.InvariantError("invalid_derivation_sources")
            derivation = list(derivation_source_uids)
            invariants.source_groups(captured, derivation)
            if (not set(mandatory).issubset(derivation)
                    or not set(derivation).issubset(self.order)):
                raise invariants.InvariantError("incomplete_known_derivation")
        uid = summary_uid(self._summary_seq + 1)
        if uid in self.summaries:
            raise invariants.InvariantError("summary_id_collision")
        self._summary_seq += 1
        anchor = min(self.messages[i].index for i in source_message_ids
                     if i in self.messages) if source_message_ids else 0
        summary = ContextSummary(
            uid=uid, content=content, source_message_ids=list(source_message_ids),
            created_by=created_by, created_at=utc_now_iso(),
            anchor_index=anchor,
            version=1, derivation_source_uids=derivation,
        )
        self._count_summary(summary)
        self.summaries[uid] = summary
        self._validate_native_presentation(archive_uids=source_message_ids)
        self._dirty = True
        return summary

    @_accepted_mutation
    def mark_archived(self, uids: List[str], *, narrative_summary_id: Optional[str] = None) -> None:
        from .policies import PolicyViolation
        resolved = self.resolve_groups(uids)
        if narrative_summary_id is None:
            self.require_edit(resolved, "archive")
        else:
            summary = self.summaries.get(narrative_summary_id)
            if (summary is None or not summary.active
                    or set(summary.source_message_ids) != set(resolved)):
                projection_refuse("projection_ineligible", "", "Narrative archival requires an active exact-membership summary.")
            self.require_edit(resolved, "narrative")
            self._validate_native_presentation(archive_uids=resolved)
        protected = [u for u in resolved if self.messages[u].protected]
        if protected:
            raise PolicyViolation("protected_context", "group contains protected context", protected)
        for uid in resolved:
            rec = self.messages[uid]
            rec.archived, rec.active = True, False
        self._dirty = True

    @_accepted_mutation
    def mark_active(self, uids: List[str]) -> None:
        resolved = self.resolve_groups(uids)
        for uid in resolved:
            rec = self.messages[uid]
            rec.archived, rec.active = False, True
            self._retire_body_compaction(rec)
            rec.token_count = self.counter.count_message(rec.envelope)
        for summary in self.summaries.values():
            if set(resolved).intersection(summary.source_message_ids):
                summary.active = False
        self._dirty = True

    @_accepted_mutation
    def compact_bodies(self, uids: List[str], content: str, created_by: str) -> List[str]:
        from .policies import PolicyViolation
        resolved = self.resolve_groups(uids)
        self.require_edit(resolved, "body")
        protected = [u for u in resolved if self.messages[u].protected]
        if protected:
            raise PolicyViolation("protected_context", "group contains protected context", protected)
        results = [u for u in resolved if self.messages[u].role == "tool"]
        if not results or any(self.messages[u].archived for u in resolved):
            raise PolicyViolation("invalid_body_compaction", "body compaction requires active tool groups", resolved)
        # A joint summary is derived from the entire resolved input batch.
        # Restoring one result still restores only its own closed tool group.
        sources_by_result = {u: list(group.source_uids)
                             for group in self.groups() for u in group.result_uids}
        derivation = invariants.derive_inputs(invariants.capture(self), resolved)
        for uid in results:
            rec = self.messages[uid]
            self._retire_body_compaction(rec)
            rec.body_compaction = {"version": 1, "content": content,
                                   "derivation_source_uids": list(derivation),
                                   "restore_group_uids": sources_by_result[uid],
                                   "created_by": created_by, "created_at": utc_now_iso()}
            wire = rec.envelope
            wire["content"] = self.render_body(rec)
            rec.token_count = self.counter.count_message(wire)
        self._dirty = True
        return resolved

    @staticmethod
    def render_body(rec: ContextMessage) -> str:
        provenance = rec.body_compaction
        return (f"[CLM BODY COMPACTION; derivation sources {', '.join(provenance['derivation_source_uids'])}; "
                f"restore group {', '.join(provenance['restore_group_uids'])}; "
                f"by {provenance['created_by']}; restore this group with clm_restore]\n"
                + provenance["content"])

    def _count_summary(self, summary: ContextSummary) -> None:
        summary.token_count = self.counter.count_message(
            provider_envelope(render_summary_message(summary, summary.anchor_index)))

    @_accepted_mutation
    def update_summary(self, summary_uid_: str, content: str,
                       created_by: str = "model") -> Optional[ContextSummary]:
        s = self.summary_by_prefix(summary_uid_)
        if s is None:
            return None
        self.require_edit(s.source_message_ids, "narrative")
        known = set(s.derivation_source_uids)
        if set(s.source_message_ids).issubset(self.order):
            known.update(invariants.derive_inputs(invariants.capture(self), s.source_message_ids))
        s.derivation_source_uids = sorted(known, key=lambda u: self.messages[u].index)
        s.content = content
        s.revision += 1
        s.created_by = created_by
        s.created_at = utc_now_iso()
        self._count_summary(s)
        self._dirty = True
        return s

    @_accepted_mutation
    def save_toolobs(self, ref: str, full_text: str) -> None:
        self.toolobs[ref] = {"content": full_text, "chars": len(full_text)}
        self._dirty = True

    def get_toolobs(self, ref: str) -> Optional[str]:
        entry = self.toolobs.get(ref)
        return entry["content"] if entry else None

    @_accepted_mutation
    def bump_revision(self) -> int:
        self.revision += 1
        self._dirty = True
        return self.revision

    # ------------------------------------------------------------------ #
    # budget                                                              #
    # ------------------------------------------------------------------ #

    def working_set_tokens(self) -> int:
        total = 0
        for rec in (self.messages[u] for u in self.order if self.messages[u].active):
            total += rec.token_count or self.counter.count_text(rec.content)
        for s in self.active_summaries():
            if self.is_summary_fully_archived(s):
                total += s.token_count or self.counter.count_text(s.content)
        return total

    # ------------------------------------------------------------------ #
    # metrics (spec §31)                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _bump(metrics: Dict[str, Any], key: str, amount: float) -> Dict[str, Any]:
        metrics = dict(metrics)
        metrics[key] = metrics.get(key, 0) + amount
        return metrics

    # ------------------------------------------------------------------ #
    # persistence (spec §29)                                              #
    # ------------------------------------------------------------------ #

    def state_dict(self, budget_tokens: int = 0, reserve_tokens: int = 0,
                   current_tokens: int = 0) -> Dict[str, Any]:
        native = self.coordinate_origin is not None or (self._origin_mode == "native" and self._origin_admission == "admitted")
        return deepcopy({
            "version": 6 if self._context_map is not None else 5 if native else _STATE_VERSION,
            **({"coordinate_origin": self.coordinate_origin,
                "origin_mode": "native" if native else "standalone",
                "context_map": self._context_map} if self._context_map is not None else
               {"coordinate_origin": self.coordinate_origin} if native else {}),
            "uid_version": 2,
            "uid_migration": self.uid_migration,
            "legacy_state": self.legacy_state,
            "recovery_required": self.recovery_required,
            "conversation_id": self.conversation_id,
            "session_id": self.session_id,
            "system_message": self.system_message,
            "system_uid": self.system_uid,
            "messages": {u: r.to_dict() for u, r in self.messages.items()},
            "order": list(self.order),
            "summaries": {u: s.to_dict() for u, s in self.summaries.items()},
            "toolobs": {u: v for u, v in self.toolobs.items()},
            "revision": self.revision,
            "summary_seq": self._summary_seq,
            "toolobs_seq": self._toolobs_seq,
            "metrics": self.metrics,
            "budget_tokens": budget_tokens,
            "reserve_tokens": reserve_tokens,
            "current_tokens": current_tokens,
        })

    def load_state_dict(self, d: Dict[str, Any], *,
                        canonical_messages: Optional[List[Dict[str, Any]]] = None,
                        legacy_summary_inputs: Optional[dict] = None,
                        context_origin_mode: str = "standalone") -> None:
        self.require_origin()
        if context_origin_mode not in {"native", "standalone"}:
            raise ValueError("unsupported context origin mode")
        self._validate_state_schema(d)
        if (context_origin_mode == "native" or self._origin_mode == "native"
                or self.coordinate_origin is not None) and (d.get("version") not in {5, 6}
                    or (d.get("version") == 6 and d.get("origin_mode") != "native")):
            projection_refuse("coordinate_resume_unproved", "", "Checkpoint has no canonical coordinate origin.")
        version = d.get("version", 1)
        if version == 1:
            candidate = ContextStore(self.config, self.counter)
            candidate.legacy_state = deepcopy(d)
            candidate.conversation_id = d.get("conversation_id", "")
            candidate.recovery_required = "legacy_envelopes_unavailable"
            if canonical_messages is not None:
                candidate._rehydrate_legacy(d, canonical_messages, legacy_summary_inputs=legacy_summary_inputs)
            if not candidate.recovery_required:
                try:
                    candidate._sync_context_map()
                except RecoveryRequired:
                    self._preserve_uncertainty(d, "context_map_provenance_unverifiable")
                    return
                candidate.recompute_protection()
                candidate.validate_accepted()
            self._restore_snapshot(candidate._snapshot())
            self._queue_context_map_backfill(version)
            return
        if version not in {2, 3, _STATE_VERSION, 5, 6} or d.get("uid_version", 2) != 2:
            raise ValueError("unsupported CLM state version")
        self._load_envelope_state(d, legacy_summary_inputs=legacy_summary_inputs)
        if context_origin_mode == "native":
            self._origin_mode = "native"
        self._queue_context_map_backfill(version)

    def _queue_context_map_backfill(self, source_version):
        """Retain a verified observation without publishing during decoding."""
        if source_version >= 6 or self.recovery_required or self._context_map is None:
            return
        data = self._context_map
        sources = {uid for ref in data['archive_refs'].values() for uid in ref['source_uids']}
        self._pending_context_map_backfill = {
            'source_state_version': source_version, 'historical_initiator': 'unknown',
            'context_map_revision': data['revision'], 'node_ids': list(data['nodes']),
            'archive_refs': list(data['archive_refs']),
            'source_uids': sorted(sources, key=lambda uid: self.messages[uid].index),
        }

    def _consume_context_map_backfill(self, state_root):
        self.require_origin()
        if self._pending_context_map_backfill is None or self.recovery_required:
            return
        from .telemetry import Telemetry
        Telemetry(self, state_root).record('context_map_backfill', initiator='host-backfill',
                                          **deepcopy(self._pending_context_map_backfill))
        self._pending_context_map_backfill = None
        self._dirty = True

    # ------------------------------------------------------------------ #
    # legacy retrofit adoption (clm.retrofit_retire_ghosts)                #
    # ------------------------------------------------------------------ #

    @classmethod
    def _retire_superseded_ghosts(cls, state, state_root, conversation_id):
        """Adopt a pre-map store: retire provably superseded legacy ghosts.

        A branch-external archived record ("ghost") left behind by a canonical
        replacement that predates the map can never be witnessed or audited —
        groups form over the branch and audits are branch-contiguous — so
        coverage refuses forever. When every ghost is exactly a superseded
        version of its branch successor (the successor's envelope is a field-
        by-field subset of the retained original), is unprotected, unreferenced
        and carries no offloaded output, the ghosts are retired from the live
        state and preserved byte-for-byte in a sidecar next to state.json.

        All-or-nothing and fail-closed: any record that does not classify
        exactly, or a sidecar write failure, returns the state untouched and
        the ordinary provenance refusal applies.
        """
        messages = state.get("messages")
        order = state.get("order")
        if not isinstance(messages, dict) or not isinstance(order, list):
            return state, None
        order_set = set(order)
        by_index = {}
        for uid, record in messages.items():
            if isinstance(record, dict):
                by_index.setdefault(record.get("index"), []).append(uid)
        ghosts = [uid for uid, record in messages.items()
                  if isinstance(record, dict) and record.get("archived")
                  and uid not in order_set]
        if not ghosts:
            return state, None
        retired = {}
        for uid in ghosts:
            record = messages[uid]
            if record.get("protected") or record.get("toolobs_ref") is not None:
                return state, None
            successors = [v for v in by_index.get(record.get("index"), [])
                          if v != uid and v in order_set]
            if len(successors) != 1:
                return state, None
            if not cls._ghost_subsumed_by_successor(record, messages[successors[0]]):
                return state, None
            if cls._ghost_structurally_referenced(state, uid):
                return state, None
            retired[uid] = deepcopy(record)
        payload = {
            "note": ("Retired by legacy retrofit adoption (clm.retrofit_retire_ghosts): "
                     "branch-external superseded versions whose retained successors carry "
                     "every field of the retired envelope. Bytes preserved here; nothing "
                     "restorable was removed."),
            "conversation_id": conversation_id,
            "retired": {uid: retired[uid] for uid in sorted(retired)},
        }
        digest = envelope_digest(payload)[:16]
        base = os.path.join(state_root, conversation_id)
        sidecar = os.path.join(base, f"retired-legacy-ghosts-{digest}.json")
        if not os.path.exists(sidecar):
            try:
                os.makedirs(base, exist_ok=True)
                fd, tmp = tempfile.mkstemp(dir=base, prefix=".retired-", suffix=".tmp")
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as f:
                        json.dump(payload, f, ensure_ascii=False)
                    os.replace(tmp, sidecar)
                except BaseException:
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass
                    raise
            except OSError:
                return state, None
        stripped = deepcopy(state)
        for uid in retired:
            del stripped["messages"][uid]
        return stripped, {"retired": sorted(retired), "sidecar": sidecar}

    @staticmethod
    def _ghost_subsumed_by_successor(ghost, successor):
        """Every field of the successor envelope is present and equal in the
        retained original; the ghost may carry extra fields (e.g. annotations
        dropped by a later normalization) but must agree on all of the
        successor's fields."""
        ghost_env = ghost.get("envelope")
        successor_env = successor.get("envelope")
        if (not isinstance(ghost_env, dict) or not isinstance(successor_env, dict)
                or ghost.get("role") != successor.get("role")):
            return False
        try:
            for key, value in successor_env.items():
                if key not in ghost_env:
                    return False
                if envelope_digest(value) != envelope_digest(ghost_env[key]):
                    return False
        except (TypeError, ValueError):
            return False
        return True

    @staticmethod
    def _ghost_structurally_referenced(state, uid):
        """Structural references outside the ghost's own record block retirement."""
        for summary in (state.get("summaries") or {}).values():
            if isinstance(summary, dict) and (
                    uid in (summary.get("source_message_ids") or [])
                    or uid in (summary.get("derivation_source_uids") or [])):
                return True
        for record in (state.get("messages") or {}).values():
            if not isinstance(record, dict):
                continue
            overlays = [record.get("body_compaction"),
                        *(record.get("body_compaction_history") or [])]
            for overlay in overlays:
                if isinstance(overlay, dict) and (
                        uid in (overlay.get("derivation_source_uids") or [])
                        or uid in (overlay.get("restore_group_uids") or [])):
                    return True
        cmap = state.get("context_map")
        if isinstance(cmap, dict):
            for witness in (cmap.get("witnesses") or {}).values():
                if isinstance(witness, dict) and uid in (witness.get("source_uids") or []):
                    return True
            for ref in (cmap.get("archive_refs") or {}).values():
                if isinstance(ref, dict) and uid in (ref.get("source_uids") or []):
                    return True
            for diag in (cmap.get("diagnostics") or {}).values():
                if isinstance(diag, dict) and (
                        uid in (diag.get("observed_source_uids") or [])
                        or uid in (diag.get("uncertified_source_uids") or [])):
                    return True
            for pres in (cmap.get("presentations") or {}).values():
                if isinstance(pres, dict) and (
                        uid == pres.get("owner_uid")
                        or uid in (pres.get("derivation_source_uids") or [])):
                    return True
            for node in (cmap.get("nodes") or {}).values():
                if isinstance(node, dict) and uid in (node.get("provenance_ids") or []):
                    return True
        migration = state.get("uid_migration") or {}
        if isinstance(migration, dict) and (uid in migration or uid in migration.values()):
            return True
        metrics = state.get("metrics") or {}
        if isinstance(metrics, dict):
            for key in metrics:
                if isinstance(key, str) and uid in key:
                    return True
        return False

    def _consume_retrofit_retirement(self, state_root):
        """Record the one-time retrofit adoption receipt with the next save."""
        self.require_origin()
        if self._pending_retrofit_retirement is None or self.recovery_required:
            return
        from .telemetry import Telemetry
        Telemetry(self, state_root).record('legacy_ghosts_retired', initiator='host-retrofit',
                                          **deepcopy(self._pending_retrofit_retirement))
        self._pending_retrofit_retirement = None
        self._dirty = True

    @staticmethod
    def _validate_state_schema(d: Dict[str, Any]) -> None:
        """Validate nested records before any migration or identity operations."""
        _object(d, "CLM state")
        version = d.get("version")
        if version is not None and (type(version) is not int or version not in {1, 2, 3, 4, 5, 6}):
            raise ValueError("unsupported CLM state version")
        if version == 6:
            if set(d) != _NATIVE_FIELDS | {"coordinate_origin", "origin_mode", "context_map"}:
                raise ValueError("missing or unknown map state fields")
            if d["origin_mode"] == "native":
                validate_coordinate_origin(d["coordinate_origin"])
            elif d["origin_mode"] != "standalone" or d["coordinate_origin"] is not None:
                raise ValueError("invalid map coordinate origin")
            _object(d["context_map"], "context_map")
        elif version == 5:
            if set(d) != _NATIVE_FIELDS | {"coordinate_origin"}:
                raise ValueError("missing or unknown native state fields")
            validate_coordinate_origin(d["coordinate_origin"])
        elif "coordinate_origin" in d:
            raise ValueError("legacy state forbids coordinate_origin")
        native = d.get("version") in {4, 5, 6} or ("version" not in d and d.get("uid_version") == 2)
        if native and version not in {5, 6} and set(d) != _NATIVE_FIELDS:
            raise ValueError("missing or unknown native state fields")
        for field_name in ("messages", "summaries", "toolobs", "metrics", "uid_migration"):
            _object(d.get(field_name, {}), field_name)
        _string_list(d.get("order", []), "order")
        _strings(d, ("conversation_id", "session_id", "system_uid", "recovery_required"),
                 nullable=("recovery_required",))
        _integers(d, ("version", "uid_version", "revision", "summary_seq", "toolobs_seq",
                      "budget_tokens", "reserve_tokens", "current_tokens"))
        if d.get("version") in {4, 5, 6} and d.get("uid_version") != 2:
            raise ValueError("missing or invalid canonical UID version")
        validate_metrics(d.get("metrics", {}))
        if d.get("legacy_state") is not None:
            _object(d["legacy_state"], "legacy_state")
        if d.get("system_message") is not None:
            _object(d["system_message"], "system_message")
        for uid, data in d.get("messages", {}).items():
            rec = ContextMessage.from_dict(data)
            if native:
                if set(data) != _NATIVE_MESSAGE_FIELDS:
                    raise ValueError("missing or unknown native message fields")
                overlays = [data["body_compaction"], *data["body_compaction_history"]]
                if any(p is not None and set(p) != _NATIVE_BODY_FIELDS for p in overlays):
                    raise ValueError("missing or unknown native body fields")
            if not uid or rec.uid != uid:
                raise ValueError("invalid message record key")
        for uid, data in d.get("summaries", {}).items():
            rec = ContextSummary.from_dict(data)
            if native and set(data) != _NATIVE_SUMMARY_FIELDS:
                raise ValueError("missing or unknown native summary fields")
            if not uid or rec.uid != uid:
                raise ValueError("invalid summary record key")
        for uid, value in d.get("uid_migration", {}).items():
            if not uid or not isinstance(value, str) or not value:
                raise ValueError("invalid UID migration")
        for entry in d.get("toolobs", {}).values():
            _object(entry, "toolobs entry")
            if native and set(entry) != {"content", "chars"}:
                raise ValueError("missing or unknown native toolobs fields")
            if not isinstance(entry.get("content"), str):
                raise ValueError("invalid toolobs content")
            _integers(entry, ("chars",))

    def _load_envelope_state(self, d: Dict[str, Any], *, legacy_summary_inputs=None) -> None:
        self.require_origin()
        self._validate_state_schema(d)
        if (self._origin_mode == "native" or self.coordinate_origin is not None) and (d.get("version") not in {5, 6}
                or (d.get("version") == 6 and d.get("origin_mode") != "native")):
            projection_refuse("coordinate_resume_unproved", "", "Checkpoint has no canonical coordinate origin.")
        if d.get("version") not in {2, 3, _STATE_VERSION, 5, 6} or d.get("uid_version", 2) != 2:
            raise ValueError("unsupported envelope state version")
        # Parse/validate in a temporary store to avoid partially adopting corrupt data.
        candidate = ContextStore(self.config, self.counter)
        candidate.coordinate_origin = deepcopy(d.get("coordinate_origin"))
        candidate._origin_mode = d["origin_mode"] if d.get("version") == 6 else self._origin_mode
        candidate.conversation_id = d.get("conversation_id", "")
        candidate.session_id = d.get("session_id", "")
        candidate.messages = {u: ContextMessage.from_dict(r) for u, r in d.get("messages", {}).items()}
        candidate.order = list(d.get("order", []))
        if len(candidate.order) != len(set(candidate.order)):
            raise ValueError("duplicate canonical records")
        for uid, rec in candidate.messages.items():
            envelope = rec.envelope
            # Persisted JSON is untrusted schema. Check shape before the
            # dictionary-only fingerprint adapter; do not mask programmer
            # AttributeErrors at the outer file-loading boundary.
            if (not isinstance(envelope, dict)
                    or not isinstance(envelope.get("role"), str) or not envelope["role"]):
                raise ValueError("canonical envelope must be an object with a role")
            if (rec.uid != uid or envelope_uid(rec.index, envelope) != uid
                    or rec.role != envelope["role"] or rec.content != _content_of(envelope)
                    or rec.active == rec.archived):
                raise ValueError("invalid canonical envelope identity")
        for uid in candidate.order:
            if uid not in candidate.messages:
                raise ValueError("missing canonical record")
        indices = [candidate.messages[u].index for u in candidate.order]
        if indices != list(range(len(indices))):
            raise ValueError("invalid canonical order")
        candidate._by_index = {candidate.messages[u].index: u for u in candidate.order}
        if d.get("version") in {_STATE_VERSION, 5, 6}:
            if (d["system_uid"] != candidate.system_uid
                    or envelope_digest(d["system_message"]) != envelope_digest(candidate.system_message)):
                raise ValueError("native system envelope correspondence mismatch")
        for uid, rec in candidate.messages.items():
            if uid not in candidate.order and rec.active:
                raise ValueError("active ghost outside canonical order")
        first_user = next((u for u in candidate.order if candidate.messages[u].role == "user"), None)
        for uid in candidate.order:
            rec = candidate.messages[uid]
            if (uid == first_user or rec.role in {"system", "developer"}) and (not rec.protected or not rec.active):
                raise ValueError("missing permanent protection")
        validation = (validate_native_protocol(candidate._group_messages()) if candidate.coordinate_origin is not None
                      else validate_tool_protocol(candidate._group_messages(), allow_pending=True))
        if not validation.valid:
            from .protocol import ProtocolError
            raise ProtocolError(validation)
        for group in candidate.groups():
            flags = {(candidate.messages[u].active, candidate.messages[u].archived) for u in group.source_uids}
            if len(flags) != 1 or (not group.complete and flags != {(True, False)}):
                raise ValueError("inconsistent tool group state")
        # Compaction is a presentation overlay. Its source references must be
        # complete and recoverable, just like summary provenance.
        uncertain_body = False
        for uid, rec in candidate.messages.items():
            provenance = rec.body_compaction
            if d.get("version") == 2:
                if rec.body_compaction_history:
                    raise ValueError("unsupported legacy overlay history")
                if provenance is not None:
                    if rec.role != "tool" or "source_uids" not in provenance:
                        raise ValueError("invalid legacy body compaction provenance")
                    old_groups = candidate._source_groups(provenance["source_uids"])
                    if (len(old_groups) != 1 or uid not in old_groups[0].result_uids
                            or provenance["source_uids"] != list(old_groups[0].source_uids)):
                        raise ValueError("incomplete legacy body compaction sources")
                    uncertain_body = True
            else:
                if provenance is not None:
                    candidate._validate_body_sources(uid, provenance, current=True)
                for historical in rec.body_compaction_history:
                    candidate._validate_body_sources(uid, historical, current=False)
        candidate.summaries = {u: ContextSummary.from_dict(s) for u, s in d.get("summaries", {}).items()}
        for summary in candidate.summaries.values():
            candidate._source_groups(summary.source_message_ids)
            sources = set(summary.source_message_ids)
            if summary.active and not sources.issubset(set(candidate.order)):
                raise ValueError("active summary from historic branch")
        candidate.toolobs = deepcopy(d.get("toolobs", {}))
        candidate.revision = d.get("revision", 0)
        candidate._summary_seq = d.get("summary_seq", 0)
        candidate._toolobs_seq = d.get("toolobs_seq", 0)
        candidate.metrics = deepcopy(d.get("metrics", {}))
        candidate.uid_migration = deepcopy(d.get("uid_migration", {}))
        candidate.legacy_state = deepcopy(d.get("legacy_state"))
        candidate.recovery_required = d.get("recovery_required")
        candidate._context_map = deepcopy(d.get("context_map"))
        if uncertain_body:
            # Old group-only metadata cannot prove a batch summary's derivation.
            # Preserve the complete uncertain state without adopting a guessed
            # view or overwriting its bytes. Complete RAW v2 states still load.
            recovery = ContextStore(self.config, self.counter)
            recovery.conversation_id = candidate.conversation_id
            recovery.session_id = candidate.session_id
            recovery.legacy_state = deepcopy(d)
            recovery.recovery_required = "legacy_body_derivation_unverifiable"
            self._restore_snapshot(recovery._snapshot())
            return
        if d.get("version") < 4 and candidate.summaries:
            if not candidate._migrate_narratives(d, legacy_summary_inputs):
                self._preserve_uncertainty(d, "legacy_narrative_derivation_unverifiable")
                return
        if candidate._context_map is None and self.config.context_map_enabled:
            try:
                candidate._sync_context_map()
            except RecoveryRequired:
                self._preserve_uncertainty(d, "context_map_provenance_unverifiable")
                return
        candidate.validate_accepted()
        candidate.recompute_protection()
        candidate.validate_accepted()
        self._restore_snapshot(candidate._snapshot())

    def _validate_body_sources(self, uid: str, overlay: dict, *, current: bool) -> None:
        invariants.validate_body(invariants.capture(self), uid, overlay, current=current)

    def _rehydrate_legacy(self, state: dict, originals: List[dict], *, legacy_summary_inputs=None) -> None:
        """Only migrate when each old record has a verifiable original witness.

        The caller supplies authoritative Hermes history, not model reconstructions.
        Verify the actual v1 UID, including its tool-call fingerprint, before
        adopting any edit flags. Missing ghost originals or ambiguous versions
        require external canonical branch recovery; mappings must be one-to-one.
        """
        self._validate_state_schema(state)
        self.recovery_required = "legacy_mapping_unverifiable"
        candidate = ContextStore(self.config, self.counter)
        candidate.conversation_id = state.get("conversation_id", "")
        candidate.session_id = state.get("session_id", "")
        try:
            # Admit the witnessed historical presentation before applying today's
            # recent window. Intrinsic root/instruction/pin checks still apply.
            candidate._ingest_messages(originals, rolling_protection=False)
            validation = validate_tool_protocol(originals, allow_pending=True)
            if not validation.valid:
                return
            mapping = {}
            for old_uid, data in state.get("messages", {}).items():
                i = data["index"]
                if type(i) is not int or i < 0 or data.get("uid") != old_uid:
                    return
                original = originals[i]
                if original.get("role") != data["role"] or _content_of(original) != data["content"]:
                    return
                if old_uid != message_uid(i, data["role"], data["content"],
                                          _tool_call_fingerprint(original)):
                    return
                ids = data.get("metadata", {}).get("tool_call_ids", [])
                if ids != [c.get("id") for c in original.get("tool_calls") or []]:
                    return
                new_uid = candidate._by_index[i]
                if new_uid in mapping.values():
                    return
                mapping[old_uid] = new_uid
                rec = candidate.messages[mapping[old_uid]]
                if rec.protected and data.get("protected") is False:
                    # Do not repair an explicit contradictory flag by guessing
                    # that the saved edit was safe. Optional absent defaults
                    # can still derive permanent protection from the witness.
                    return
                rec.active, rec.archived = data.get("active", True), data.get("archived", False)
                rec.metadata.update(deepcopy(data.get("metadata", {})))
                # Root/pin reasons can be proven permanent; recent reasons stay
                # legacy temporary state for Task 2's recomputation.
                reason = rec.metadata.get("protect_reason")
                rec.protected = data.get("protected", False) or rec.protected
                if reason in {"root_task", "pinned"} and reason not in rec.permanent_reasons:
                    rec.permanent_reasons.append(reason)
                if reason in {"recent_user", "recent_assistant"}:
                    rec.temporary_reasons.append(reason)
            system = state.get("system_message")
            if system:
                matches = [i for i, m in enumerate(originals)
                           if envelope_digest(m) == envelope_digest(system)]
                if len(matches) != 1:
                    return
                i = matches[0]
                old_uid = state.get("system_uid")
                if (old_uid != message_uid(i, system["role"], _content_of(system),
                                           _tool_call_fingerprint(system))
                        or old_uid in mapping or candidate._by_index[i] in mapping.values()):
                    return
                mapping[old_uid] = candidate._by_index[i]
            old_order = state.get("order", [])
            if (len(old_order) != len(set(old_order))
                    or set(old_order) != set(state.get("messages", {}))):
                # A historical branch cannot be recovered from the current
                # branch alone. Keep the complete v1 state for external replay.
                return
            candidate.summaries = {u: ContextSummary.from_dict(s) for u, s in state.get("summaries", {}).items()}
            for summary in candidate.summaries.values():
                summary.source_message_ids = [mapping[u] for u in summary.source_message_ids]
            for old_uid, data in state.get("messages", {}).items():
                candidate.messages[mapping[old_uid]].summary_parent_ids = list(data.get("summary_parent_ids", []))
            # Legacy partial archives must not be advertised as a valid resume.
            for group in candidate.groups():
                if len({candidate.messages[u].active for u in group.source_uids}) != 1:
                    return
            candidate.conversation_id = state.get("conversation_id", "")
            candidate.session_id = state.get("session_id", "")
            candidate.revision = state.get("revision", 0)
            candidate._summary_seq = state.get("summary_seq", 0)
            candidate._toolobs_seq = state.get("toolobs_seq", 0)
            candidate.toolobs = deepcopy(state.get("toolobs", {}))
            candidate.metrics = deepcopy(state.get("metrics", {}))
            candidate.uid_migration = mapping
            candidate.legacy_state = deepcopy(state)
            for summary in candidate.summaries.values():
                candidate._source_groups(summary.source_message_ids)
            if candidate.summaries and not candidate._migrate_narratives(state, legacy_summary_inputs):
                self.recovery_required = "legacy_narrative_derivation_unverifiable"
                return
            candidate._sync_context_map()
            candidate.validate_accepted()
            self._restore_snapshot(candidate._snapshot())
        except (KeyError, IndexError, TypeError, ValueError):
            return

    def _preserve_uncertainty(self, state: dict, code: str) -> None:
        recovery = ContextStore(self.config, self.counter)
        recovery.conversation_id = state.get("conversation_id", "")
        recovery.session_id = state.get("session_id", "")
        recovery.legacy_state = deepcopy(state)
        recovery.recovery_required = code
        self._restore_snapshot(recovery._snapshot())

    def _migrate_narratives(self, state: dict, witness: Optional[dict]) -> bool:
        """Validate explicit host evidence; saved JSON never supplies authority.

        Historic inputs may no longer be attached. Their captured payloads and
        exact originals must still prove schema, joins and branch closure.
        This attests host-known inputs, not cryptographic authenticity or prose.
        """
        def keys(value, expected):
            if type(value) is not dict or set(value) != set(expected):
                raise ValueError("invalid legacy input witness fields")

        def translate(values, *, closed=True):
            if type(values) is not list:
                raise ValueError("witness join must be a list")
            _string_list(values, "witness join")
            if not values or len(values) != len(set(values)):
                raise ValueError("empty or duplicate witness join")
            result = [self.uid_migration[u] if state.get("version", 1) == 1 else u for u in values]
            if any(u not in self.messages for u in result):
                raise ValueError("unavailable witness original")
            indices = [self.messages[u].index for u in result]
            if indices != sorted(set(indices)):
                raise ValueError("unordered or mixed witness branch")
            if closed:
                self._source_groups(result)
            return result

        try:
            keys(witness, ("version", "legacy_state_digest", "summaries"))
            if type(witness["version"]) is not int or witness["version"] != 1:
                raise ValueError("invalid witness version")
            if type(witness["legacy_state_digest"]) is not str or witness["legacy_state_digest"] != envelope_digest(state):
                raise ValueError("legacy witness digest mismatch")
            entries = witness["summaries"]
            if type(entries) is not list or len(entries) != len(self.summaries):
                raise ValueError("incomplete legacy summary witness")
            migrated = {}
            for entry in entries:
                keys(entry, ("uid", "revision", "consumed_input_uids", "original_envelopes", "consumed_presentations"))
                uid = entry["uid"]
                if type(uid) is not str or uid not in self.summaries or uid in migrated:
                    raise ValueError("unknown or duplicate witness summary")
                target = self.summaries[uid]
                revision = entry["revision"]
                if type(revision) is not int or revision < 1 or revision != target.revision:
                    raise ValueError("legacy witness revision mismatch")
                derivation = translate(entry["consumed_input_uids"])
                originals = entry["original_envelopes"]
                keys(originals, entry["consumed_input_uids"])
                for old, new in zip(entry["consumed_input_uids"], derivation):
                    if (type(originals[old]) is not dict
                            or envelope_digest(originals[old]) != envelope_digest(self.messages[new].envelope)):
                        raise ValueError("legacy witness original mismatch")
                if not set(target.source_message_ids).issubset(derivation):
                    raise ValueError("legacy membership outside inputs")
                presentations = entry["consumed_presentations"]
                if type(presentations) is not list or not presentations:
                    raise ValueError("missing captured presentations")
                flattened = set()
                identities = set()
                for p in presentations:
                    keys(p, ("kind", "uid", "revision", "membership_uids", "derivation_source_uids", "payload"))
                    kind, identity, rev, payload = p["kind"], p["uid"], p["revision"], p["payload"]
                    if type(kind) is not str or kind not in {"raw", "body", "narrative"} or type(identity) is not str or not identity or type(payload) is not dict:
                        raise ValueError("invalid captured presentation")
                    signature = (kind, identity, rev, envelope_digest(payload))
                    if signature in identities:
                        raise ValueError("duplicate captured presentation")
                    identities.add(signature)
                    membership = translate(p["membership_uids"], closed=kind != "raw")
                    inputs = translate(p["derivation_source_uids"], closed=kind != "raw")
                    if not set(membership).issubset(inputs):
                        raise ValueError("captured membership outside inputs")
                    if kind == "raw":
                        if rev is not None or p["membership_uids"] != [identity] or p["derivation_source_uids"] != [identity]:
                            raise ValueError("invalid raw capture")
                        if envelope_digest(payload) != envelope_digest(self.messages[membership[0]].envelope):
                            raise ValueError("captured raw envelope mismatch")
                    elif kind == "body":
                        if rev is not None:
                            raise ValueError("body version is not a revision")
                        original = translate([identity], closed=False)[0]
                        validate_body_compaction(payload)
                        if state.get("version") == 3:
                            keys(payload, ("version", "content", "derivation_source_uids", "restore_group_uids", "created_by", "created_at"))
                            self._validate_body_sources(original, payload, current=False)
                            if payload["restore_group_uids"] != p["membership_uids"] or payload["derivation_source_uids"] != p["derivation_source_uids"]:
                                raise ValueError("captured body joins mismatch")
                        else:
                            keys(payload, ("content", "source_uids", "created_by", "created_at"))
                            if "source_uids" not in payload or payload["source_uids"] != p["membership_uids"]:
                                raise ValueError("captured legacy body membership mismatch")
                        groups = self._source_groups(membership)
                        if len(groups) != 1 or original not in groups[0].result_uids or list(groups[0].source_uids) != membership:
                            raise ValueError("invalid captured body group")
                    else:
                        if type(rev) is not int or rev < 1:
                            raise ValueError("invalid captured narrative revision")
                        keys(payload, ("uid", "content", "source_message_ids", "created_by", "created_at",
                                       "token_count", "revision", "anchor_index", "active"))
                        captured = ContextSummary.from_dict(payload)
                        if captured.uid != identity or captured.revision != rev or captured.source_message_ids != p["membership_uids"]:
                            raise ValueError("captured narrative identity mismatch")
                        if captured.anchor_index != self.messages[membership[0]].index:
                            raise ValueError("captured narrative anchor mismatch")
                        saved = state.get("summaries", {}).get(identity)
                        if (saved is not None and saved.get("revision", 1) == rev
                                and envelope_digest(saved) != envelope_digest(payload)):
                            raise ValueError("retained narrative payload mismatch")
                        if saved is not None and rev > saved.get("revision", 1):
                            raise ValueError("future captured narrative revision")
                    flattened.update(inputs)
                if flattened != set(derivation):
                    raise ValueError("captured input union mismatch")
                migrated[uid] = derivation
            if set(migrated) != set(self.summaries):
                raise ValueError("missing legacy narrative witness")
            for uid, derivation in migrated.items():
                s = self.summaries[uid]
                s.version, s.derivation_source_uids = 1, derivation
                self._count_summary(s)
            return True
        except (KeyError, TypeError, ValueError):
            return False

    def retry_save_rollback(self) -> bool:
        """Restore the original standalone save preimage before reopening admission."""
        owner = self._pending_save_rollback
        if owner is None:
            return False
        # Keep the owner outside logical snapshots and held throughout restore.
        # Any failure leaves this same preimage available for the next retry.
        try:
            self._restore_snapshot(owner["snapshot"])
            self.counter.__dict__ = deepcopy(owner["counter"])
        finally:
            self._restore_session_files(owner["files"])
        self._pending_save_rollback = None
        return True

    def save(self, state_root: str, budget_tokens: int = 0, reserve_tokens: int = 0,
             current_tokens: int = 0) -> Optional[str]:
        """Atomically write state.json under <state_root>/<conversation_id>/."""
        self.require_origin()
        if not self.conversation_id or self.recovery_required:
            return None
        self._validate_conversation_identity(self.conversation_id)
        self.validate_accepted()
        # A standalone save owns pending reconstruction publication. An existing
        # engine/store transaction keeps its buffer and its own rollback owner.
        owns_backfill = self._pending_context_map_backfill is not None and self._event_buffer is None
        snapshot = self._snapshot() if owns_backfill else None
        counter = deepcopy(vars(self.counter)) if owns_backfill else None
        files = self._snapshot_session_files(state_root, self.conversation_id) if owns_backfill else None
        try:
            if owns_backfill:
                self.begin_transaction()
            self._consume_context_map_backfill(state_root)
            self._consume_retrofit_retirement(state_root)
            path = self._save_state(state_root, budget_tokens, reserve_tokens, current_tokens)
            if owns_backfill:
                self.commit_transaction(state_root, require_event_publication=True)
            return path
        except BaseException:
            if owns_backfill:
                self._pending_save_rollback = {"snapshot": snapshot, "counter": counter, "files": files}
                self.retry_save_rollback()
            raise

    def _save_state(self, state_root, budget_tokens, reserve_tokens, current_tokens):
        payload = self.state_dict(budget_tokens, reserve_tokens, current_tokens)
        self._validate_state_schema(payload)
        # Validate exactly the detached bytes that will be written. A caller may
        # override state_dict; validating the live receiver cannot attest that export.
        exported = ContextStore(self.config)
        exported.load_state_dict(payload, context_origin_mode=(
            "native" if self.coordinate_origin is not None else self._origin_mode))
        exported.validate_accepted()
        base = os.path.join(state_root, self.conversation_id)
        os.makedirs(base, exist_ok=True)
        path = os.path.join(base, "state.json")
        fd, tmp = tempfile.mkstemp(dir=base, prefix=".state-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        self._dirty = False
        return path

    @classmethod
    def load(cls, state_root: str, conversation_id: str, config: Any,
             counter: Optional[TokenCounter] = None, *,
             canonical_messages: Optional[List[dict]] = None,
             legacy_summary_inputs: Optional[dict] = None,
             context_origin_mode: str = "standalone") -> "ContextStore":
        if context_origin_mode not in {"native", "standalone"}:
            raise ValueError("unsupported context origin mode")
        cls._validate_conversation_identity(conversation_id)
        store = cls(config, counter)
        store.conversation_id = conversation_id
        path = os.path.join(state_root, conversation_id, "state.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    def unique_object(pairs):
                        result = {}
                        for key, value in pairs:
                            if key in result:
                                raise ValueError("duplicate state JSON key")
                            result[key] = value
                        return result
                    state = json.load(f, object_pairs_hook=unique_object)
                    if (type(state) is not dict or type(state.get("conversation_id")) is not str
                            or not state["conversation_id"] or state["conversation_id"] != conversation_id):
                        raise ValueError("state conversation ownership mismatch")
                    retirement = None
                    if (getattr(config, "context_map_enabled", False)
                            and getattr(config, "retrofit_retire_ghosts", False)):
                        state, retirement = cls._retire_superseded_ghosts(
                            state, state_root, conversation_id)
                    store.load_state_dict(state, canonical_messages=canonical_messages,
                                          legacy_summary_inputs=legacy_summary_inputs,
                                          context_origin_mode=context_origin_mode)
                    if retirement is not None and store.recovery_required is None:
                        store._pending_retrofit_retirement = retirement
            except (OSError, ValueError, KeyError, TypeError) as error:
                if type(error).__name__ == "SourceProjectionError":
                    raise
                store.recovery_required = "state_unavailable"
        if context_origin_mode == "native":
            store._origin_mode = "native"
            store._origin_admission = "pending"
            store._origin_reason = {"code": "coordinate_resume_unproved", "path": "",
                                    "message": "Candidate requires native lifecycle admission."}
        return store

    def append_event(self, state_root: str, event: Dict[str, Any]) -> None:
        self.require_origin()
        if not self.config.event_log or not self.conversation_id:
            return
        self._validate_conversation_identity(self.conversation_id)
        if self._event_buffer is not None:
            self._event_buffer.append(deepcopy(event))
            return
        base = os.path.join(state_root, self.conversation_id)
        try:
            os.makedirs(base, exist_ok=True)
            with open(os.path.join(base, "events.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except OSError:
            if self._event_publication_required:
                raise
