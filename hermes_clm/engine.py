"""CLM context engine (spec §5, §37, §42 Phase 0/1).

Hermes ``ContextEngine`` plugin that lets the model operate on a reversible,
structured working-context layer while the canonical Hermes conversation stays
intact underneath (spec §60).

Host contract (agent/context_engine.py):
  * ``select_context`` — per-request, request-only replacement (never mutates
    the persisted transcript);
  * ``on_turn_complete`` — observe finished turns to ingest/index;
  * ``get_tool_schemas`` / ``handle_tool_call`` — the five CLM tools (spec §9);
  * ``update_from_response`` — real provider usage for calibration;
  * ``clone_for_agent`` — per-agent instance; parent/subagent isolation is the
    host's job (each agent gets its own clone + session id).
"""

from __future__ import annotations

import json
import logging
import os
import time
from copy import deepcopy
from functools import wraps
from contextlib import nullcontext
from typing import Any, Dict, List, Optional

from .budget import ContextBudgetController
from .config import CLMConfig, resolve_state_root
from .models import utc_now_iso
from .policies import PolicyViolation, emergency_order
from .prompts import CLM_SYSTEM_GUIDANCE
from .selector import ContextSelector
from .store import ContextStore
from .telemetry import Telemetry, observe_reference_restore, restore_timestamps
from .tokenizer import TokenCounter
from .protocol import (envelope_uid, envelope_digest, provider_envelope, validate_tool_protocol,
                       ProtocolError, RecoveryRequired, COORDINATE_ORIGIN,
                       projection_refuse, validate_source_projection, validate_source_conflicts,
                       validate_json)
from . import invariants
from .retrieval.context_map import _OMITTED_LIMIT
from .retrieval.restore import OMITTED as _OMITTED_RESTORE_ARGUMENT, plan_reference_restore

logger = logging.getLogger(__name__)
_OMITTED_MAP_METADATA = object()

try:  # in-host (plugin loaded by Hermes) vs standalone (tests / benchmarks)
    from agent.context_engine import ContextEngine as _HostBase
except Exception:  # pragma: no cover - standalone fallback
    _HostBase = None  # type: ignore


def _content_chars(message: Dict[str, Any]) -> int:
    content = message.get("content")
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(len(str(p.get("text", ""))) for p in content
                   if isinstance(p, dict) and p.get("type") == "text")
    return 0 if content is None else len(str(content))


def _current_protection(operation=None, *, budget_edit=True):
    """Refresh current configuration before policy checks, with full rollback.

    Each operation retains its existing persistence transaction. This outer
    snapshot also owns the derived changes preceding those admission checks.
    """
    if operation is None:
        return lambda target: _current_protection(target, budget_edit=budget_edit)

    @wraps(operation)
    def apply(self, *args, **kwargs):
        self._require_origin()
        self.store.validate_accepted()
        snapshot = self._transaction_snapshot()
        files = None
        try:
            # Native tool dispatch owns its deferred publication separately.
            # Independently called edits own any state/events they publish here.
            if (self._pending_tool_budget is None and self._conversation_id
                    and (self.config.persistence or self.config.event_log)):
                files = self.store._snapshot_session_files(self._state_root, self._conversation_id)
            self.store.recompute_protection()
            with (self._exact_edit_scope(operation.__name__) if budget_edit else nullcontext()):
                result = operation(self, *args, **kwargs)
                if self.config.budget_mode == 'exact' and isinstance(result, dict) and result.get('ok'):
                    result['final_budget'] = deepcopy(self._last_budget_decision)
                return result
        except BaseException:
            self._rollback_direct_edit(snapshot, files)
            raise
    return apply


from .exact_budget import ExactBudgetMixin
from .tool_budget import ToolBudgetMixin, _MUTATING_TOOLS
from .map_snapshot import MapSnapshotMixin
from .native_replay import NativeReplayMixin
from .recovery import RecoveryTurnMixin


class CLMEngine(NativeReplayMixin, MapSnapshotMixin, ExactBudgetMixin, ToolBudgetMixin, RecoveryTurnMixin, _HostBase or object):  # type: ignore[misc, arg-type]
    """Model-directed context engine (see module docstring)."""

    # ------------------------------------------------------------------ #
    # host-facing identity + token state (run_agent.py reads these)      #
    # ------------------------------------------------------------------ #

    @property
    def name(self) -> str:
        return "clm"

    last_prompt_tokens = 0
    last_completion_tokens = 0
    last_total_tokens = 0
    threshold_tokens = 0
    context_length = 0
    compression_count = 0
    threshold_percent = 0.75
    protect_first_n = 3
    protect_last_n = 6
    emit_automatic_compaction_status = False

    # ------------------------------------------------------------------ #
    # construction                                                        #
    # ------------------------------------------------------------------ #

    def __init__(self, config: Optional[CLMConfig] = None,
                 hermes_home: str = ""):
        super().__init__()
        self.config = config or CLMConfig.from_hermes(None, hermes_home)
        self.hermes_home = hermes_home
        self.counter = TokenCounter(self.config.tokenizer)
        self._state_root = resolve_state_root(self.config, hermes_home)
        self.store = ContextStore(self.config, self.counter)
        self.budget = ContextBudgetController(self.config)
        self.selector = ContextSelector(self.store, self.budget)
        self.telemetry = Telemetry(self.store, self._state_root)
        self._session_id = ""
        self._conversation_id = ""
        self._overflow = False
        self._overflow_retries = 0
        self._last_working_tokens: Optional[int] = None
        self._pending_estimate: Optional[Dict[str, Any]] = None
        self._origin_mode = "standalone"
        self._origin_admission = "admitted"
        self._origin_reason = None
        self._lookup_receipts = []
        self._last_rendered_messages: Optional[List[Dict[str, Any]]] = None
        self._last_rendered_bytes: Optional[bytes] = None
        self._last_render_diagnostic: Optional[Dict[str, Any]] = None
        self._render_diagnostics: List[Dict[str, Any]] = []
        self._initialize_exact_budget()
        self._initialize_tool_budget()
        self._initialize_recovery_turn()

    def _attach_store(self, store: ContextStore) -> None:
        """Re-point selector/telemetry at a (re)loaded store."""
        restore_observer = self.telemetry._reference_restores
        self.store = store
        self.selector = ContextSelector(store, self.budget)
        self.telemetry = Telemetry(store, self._state_root)
        self.telemetry._reference_restores = restore_observer
        store._budget_validator = self._validate_store_budget
        store._transaction_admission = self._require_context_rollback_complete

    # ------------------------------------------------------------------ #
    # session lifecycle                                                   #
    # ------------------------------------------------------------------ #

    def prepare_context_origin(self, mode: str) -> None:
        if type(mode) is not str or mode not in {"native", "standalone"}:
            raise ValueError("unsupported context origin mode")
        if self._origin_mode == "native":
            if mode != "native":
                projection_refuse("coordinate_resume_unproved", "", "Native origin cannot be downgraded.")
            return
        if mode == "native":
            self._origin_mode = "native"
            self._origin_admission = "pending"
            self.store._origin_mode = "native"
            self.store._origin_admission = "pending"

    def _require_origin(self) -> None:
        self.store.require_origin()

    def _admit_native_session(self, session_id, kwargs):
        if self._origin_admission == "refused":
            self._require_origin()
        conversation_id = kwargs["conversation_id"]
        if self._origin_admission == "admitted" and conversation_id == self._conversation_id:
            return
        path = os.path.join(self._state_root, conversation_id, "state.json")
        exists = os.path.exists(path)
        try:
            candidate = ContextStore.load(self._state_root, conversation_id, self.config,
                                          context_origin_mode="native") if exists else ContextStore(self.config, self.counter)
        except ValueError as error:
            if type(error).__name__ != "SourceProjectionError":
                raise
            candidate = ContextStore(self.config, self.counter)
            candidate.recovery_required = "coordinate_resume_unproved"
        if (candidate.recovery_required or (exists and candidate.coordinate_origin is None)
                or (not exists and (self.store.messages or self.store.summaries or self.store.toolobs
                                   or self._conversation_id))):
            reason = {"code": "coordinate_resume_unproved", "path": "",
                      "message": "Native target has no proven canonical coordinate lineage."}
            self._origin_admission = "refused"
            self._origin_reason = deepcopy(reason)
            self.store._origin_admission = "refused"
            self.store._origin_reason = deepcopy(reason)
            projection_refuse(**reason)
        candidate.coordinate_origin = deepcopy(COORDINATE_ORIGIN)
        candidate._origin_mode = "native"
        candidate._origin_admission = "admitted"
        candidate._origin_reason = None
        candidate.counter = self.counter
        candidate.conversation_id = conversation_id
        candidate.session_id = session_id or ""
        self._origin_admission = "admitted"
        self._origin_reason = None
        self._attach_store(candidate)

    def on_session_start(self, session_id: str, **kwargs: Any) -> None:
        self._require_context_rollback_complete()
        previous_session = kwargs.pop("previous_session", None)
        prior_engine = previous_session["engine"] if previous_session is not None else None
        if prior_engine is not None and (not isinstance(prior_engine, CLMEngine) or prior_engine is self):
            raise ValueError("Native session transition requires a distinct compatible prior engine")
        if "context_origin_mode" in kwargs:
            self.prepare_context_origin(kwargs["context_origin_mode"])
        # Resolve once, before admission reads, file snapshots or prior-session
        # callbacks. Every startup owner uses this exact validated component.
        target = kwargs.get("conversation_id") or kwargs.get("task_id") or session_id or "default"
        try:
            ContextStore._validate_conversation_identity(target)
        except ValueError:
            if self._origin_mode == "native":
                projection_refuse("coordinate_resume_unproved", "", "Unsafe native target identity.")
            raise
        kwargs["conversation_id"] = target
        previous = self.store
        previous_services = {key: getattr(self, key) for key in
                             ("store", "selector", "telemetry", "counter", "budget")}
        # A prepared native engine is not yet admitted. Capturing its preimage
        # must not grant origin authority merely to own startup recovery.
        snapshot = self._capture_transaction_snapshot()
        if self._origin_mode == "native":
            self._admit_native_session(session_id, kwargs)
        prior_snapshot = None
        files, prior_files = None, None
        try:
            self.store.begin_transaction()
            if self.config.persistence or self.config.event_log:
                files = self.store._snapshot_session_files(self._state_root, target)
            if prior_engine is not None:
                prior_engine._require_origin()
                prior_snapshot = prior_engine._transaction_snapshot()
                if prior_engine.config.persistence or prior_engine.config.event_log:
                    prior_files = prior_engine.store._snapshot_session_files(
                        prior_engine._state_root, prior_engine.store.conversation_id)
                prior_engine.on_session_end(previous_session["session_id"], previous_session["messages"])
                if (self._origin_mode == "native"
                        and os.path.realpath(self._state_root) == os.path.realpath(prior_engine._state_root)
                        and self.store.conversation_id == prior_engine.store.conversation_id):
                    # Admission read the old checkpoint before the final tail was
                    # observed. A same-conversation transition must retain that tail.
                    self.store.load_state_dict(prior_engine.store.state_dict(), context_origin_mode="native")
            self._start_session(session_id, **kwargs)
            self._publish_context_transaction()
            previous._event_buffer = None
        except BaseException:
            owners = []
            if prior_snapshot is not None:
                owners.append((prior_engine, prior_snapshot, prior_files, None))
            owners.append((self, snapshot, files, previous_services))
            shared = (files is not None and prior_files is not None
                      and {os.path.realpath(p) for p in files['files']}
                      & {os.path.realpath(p) for p in prior_files['files']})
            if shared:
                shared_owners = tuple(owner for owner, *_ in owners)
                for owner, preimage, owned_files, services in owners:
                    owner._retain_direct_rollback(preimage, owned_files, services, shared_owners)
                self.retry_context_rollback()
            else:
                failure = None
                for owner, preimage, owned_files, services in owners:
                    try:
                        owner._rollback_direct_edit(preimage, owned_files, services)
                    except BaseException as error:
                        if failure is None:
                            failure = error
                if failure is not None:
                    raise failure
            raise

    def _start_session(self, session_id: str, **kwargs: Any) -> None:
        self._lookup_receipts = []
        self._session_id = session_id or ""
        self._conversation_id = kwargs["conversation_id"]
        if self.config.persistence and self._origin_mode != "native":
            loaded = ContextStore.load(
                self._state_root, self._conversation_id, self.config, self.counter,
                canonical_messages=kwargs.get("canonical_messages"),
                legacy_summary_inputs=kwargs.get("legacy_summary_inputs"))
            # Adopt the loaded store's token calibration if it was saved warm.
            self._attach_store(loaded)
            self.store._event_buffer = []
            if self.store.recovery_required:
                return
        else:
            self.store.conversation_id = self._conversation_id
        self.store.session_id = self._session_id
        self.store._projection_policy = {}
        self.budget.provider_context_length = max(
            self.budget.provider_context_length,
            int(kwargs.get("context_length") or 0))
        self.budget.active_target = self.config.active_target_tokens
        self.budget.reset_nudges()
        self.budget.reset_overflow_retries()
        self._overflow = False
        self._overflow_retries = 0
        self._last_working_tokens = None
        self._recovery_turn.reset()
        self.store._consume_context_map_backfill(self._state_root)
        self.store._consume_retrofit_retirement(self._state_root)
        self.telemetry.record("session_start", initiator="host",
                              session_id=self._session_id,
                              conversation_id=self._conversation_id)
        self._persist()

    def on_session_end(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        self._require_origin()
        snapshot = self._begin_transaction()
        files = None
        try:
            if self._conversation_id and (self.config.persistence or self.config.event_log):
                files = self.store._snapshot_session_files(self._state_root, self._conversation_id)
            if messages:
                self._ingest(self._align_host_observer_messages(list(messages)))
            self.telemetry.record("session_end", initiator="host",
                                  metrics=self.telemetry.snapshot())
            self._persist()
            self._publish_context_transaction()
        except BaseException as exc:
            self._rollback_direct_edit(snapshot, files)
            if not isinstance(exc, Exception):
                raise
            logger.debug("CLM on_session_end failed", exc_info=True)
            if isinstance(exc, OSError) or self._origin_mode == "native":
                raise

    @_current_protection(budget_edit=False)
    def on_session_reset(self) -> None:
        self._require_origin()
        self._lookup_receipts = []
        self.store._projection_policy = {}
        self.store._projection_context = None
        self.store.recompute_protection()
        super().on_session_reset()
        self.compression_count = 0
        self.budget.reset_nudges()
        self.budget.reset_overflow_retries()
        self._overflow = False
        self._overflow_retries = 0
        self._last_working_tokens = None
        self._pending_estimate = None
        self._budget_frame = None
        self._last_budget_decision = None
        self._exact_pressure_notice = None
        self._pressure_episode.reset()
        self._recovery_turn.reset()
        self._budget_excursion = None
        self.telemetry.record("session_reset", initiator="host")

    def clone_for_agent(self) -> "CLMEngine":
        """Per-agent instance (spec §25). The host calls on_session_start for
        each agent, which (re)loads the per-agent persisted state keyed by the
        agent's conversation id — parent/subagent states are isolated."""
        clone = CLMEngine(self.config, self.hermes_home)
        clone.counter._samples = list(self.counter._samples)
        clone.counter._ratio = self.counter._ratio
        if self._exact_counter is not None:
            clone.bind_exact_counter(self._exact_counter, self._budget_identity)
            clone._budget_binding_invalid = self._budget_binding_invalid
        return clone

    def bind_session_state(self, session_db: Any = None, session_id: str = "") -> None:
        """Detach persistence and event logging when bound to an ephemeral review fork."""
        if session_db is None and session_id == "":
            if (self._conversation_id and not self.store.messages
                    and not self.store.summaries and self._state_root):
                import shutil
                orphan_dir = os.path.join(self._state_root, self._conversation_id)
                if os.path.isdir(orphan_dir):
                    shutil.rmtree(orphan_dir, ignore_errors=True)
            self.config = deepcopy(self.config)
            self.config.persistence = False
            self.config.event_log = False
            self.store.config = self.config
            self.budget.config = self.config

    def adopt_detached_parent(self, parent_engine: Any) -> None:
        """Inherit the parent engine's in-memory working set on a detached fork for cache parity."""
        if not isinstance(parent_engine, CLMEngine):
            return
        self.config = deepcopy(parent_engine.config)
        self.config.persistence = False
        self.config.event_log = False
        self.counter = deepcopy(parent_engine.counter)
        self.budget = deepcopy(parent_engine.budget)
        self.budget.config = self.config
        cloned_store = deepcopy(parent_engine.store)
        cloned_store.config = self.config
        cloned_store.counter = self.counter
        self._attach_store(cloned_store)
        self._session_id = parent_engine._session_id
        self._conversation_id = parent_engine._conversation_id
        self._overflow = parent_engine._overflow
        self._overflow_retries = parent_engine._overflow_retries
        self._last_working_tokens = parent_engine._last_working_tokens

    # ------------------------------------------------------------------ #
    # host token accounting                                               #
    # ------------------------------------------------------------------ #

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        self._require_origin()
        usage = usage or {}
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
        total = int(usage.get("total_tokens") or 0) or (prompt + completion)
        self.last_prompt_tokens = prompt
        self.last_completion_tokens = completion
        self.last_total_tokens = total
        self.telemetry.model_call_metrics(prompt, completion)
        pending = self._pending_estimate
        self._pending_estimate = None
        if pending and prompt > 100 and self.config.budget_mode == "exact":
            # In approximate host mode, working_set_tokens and usable/hard budgets
            # operate in consistent engine-estimated currency while prompt_tokens
            # includes un-reclaimable wire overhead (system prompts + tool schemas).
            # Only calibrate the heuristic counter in exact budget mode.
            system_tokens = (self.counter.count_message(self.store.system_message)
                             if self.store.system_message else 0)
            real_content = max(1, prompt - system_tokens)
            self.counter.calibrate(pending["tokens"], real_content,
                                   pending["chars"], pending["chars"])
        self.compression_count = int(
            self.store.metrics.get("context_compactions", 0))

    def update_model(self, model: str, context_length: int, base_url: str = "",
                     api_key: str = "", provider: str = "", api_mode: str = "") -> None:
        self._require_origin()
        if self.config.budget_mode == "exact":
            self.context_length = int(context_length or 0)
            self.budget.provider_context_length = self.context_length
            identity = self._budget_identity
            if identity is not None and (
                    model != identity.model or self.context_length != identity.context_tokens
                    or (base_url and base_url.rstrip('/') != identity.base_url.rstrip('/'))
                    or (provider and provider != identity.provider)
                    or (api_mode and api_mode != "chat_completions")):
                self._budget_binding_invalid = True
                self._budget_frame = None
            self.threshold_tokens = self.config.usable_target_tokens
            return
        if _HostBase is not None:
            super().update_model(model, context_length, base_url=base_url,
                                 api_key=api_key, provider=provider,
                                 api_mode=api_mode)
        else:
            self.context_length = int(context_length or 0)
        # The provider window is safety headroom, never the working budget.
        self.budget.provider_context_length = max(
            self.budget.provider_context_length, self.context_length,
            self.config.provider_context_min_tokens)
        # threshold_tokens = the hard working limit (run_agent.py preflight
        # reads this; should_compress() is False so it never auto-fires).
        self.threshold_tokens = int(self.budget.hard_limit)
        self.threshold_percent = self.config.overflow_fraction_of_provider

    # ------------------------------------------------------------------ #
    # compression hooks (CLM owns policy; host auto-compression stays off)#
    # ------------------------------------------------------------------ #

    def should_compress(self, prompt_tokens: Optional[int] = None) -> bool:
        return False

    def should_compress_info(self, prompt_tokens: Optional[int] = None):
        return False, ("CLM manages the working set itself "
                       "(select_context + overflow controller)")

    def has_content_to_compress(self, messages: List[Dict[str, Any]]) -> bool:
        self._require_origin()
        try:
            self._ingest(self._align_host_observer_messages(list(messages or [])))
        except Exception:
            logger.debug("CLM has_content_to_compress ingest failed",
                         exc_info=True)
        return any(
            (rec := self.store.get(uid)) is not None and not rec.protected
            for uid in self.store.order
        )

    def compress(self, messages: List[Dict[str, Any]], current_tokens: Optional[int] = None,
                 focus_topic: Optional[str] = None, force: bool = False,
                 memory_context: str = "") -> List[Dict[str, Any]]:
        """Host /compress: deterministic emergency pass (spec §19), then return
        the rebuilt working set (request-only, never persisted)."""
        self._require_origin()
        result = self._emergency_compress(initiator="host-manual")
        notice = ("CLM emergency compaction completed (host-initiated)."
                  if result and result.get("removed") else None)
        return self.selector.build(notice_text=notice)

    # ------------------------------------------------------------------ #
    # per-turn observation                                                #
    # ------------------------------------------------------------------ #

    def on_turn_complete(self, messages: List[Dict[str, Any]],
                         usage: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        self._require_origin()
        if isinstance(usage, dict) and usage:
            # Host-reported true prompt size of the completed request: bridge it
            # into accounting so production arming works on the current host,
            # which provides usage but never calls bind_exact_counter.
            try:
                self.update_from_response(usage)
            except Exception:
                logger.warning("CLM usage bridge failed", exc_info=True)
        # Bounded recovery turn: the turn boundary settles the offered turn before
        # the finished turn is ingested, while the accepted exact frame still
        # matches the store (engine-initiated edits are gated on that match).
        settled_recovery = None
        if self._recovery_turn_enabled() and self._recovery_turn.state == 'offered':
            try:
                settled_recovery = self._settle_recovery_turn(trigger='turn_complete')
            except BaseException as exc:
                if not isinstance(exc, Exception):
                    raise
                logger.warning("CLM recovery turn settlement failed", exc_info=True)
                if isinstance(exc, OSError) or (self._origin_mode == "native" and type(exc).__name__ == "SourceProjectionError"):
                    raise
        snapshot = self._begin_transaction()
        files = None
        try:
            if self._conversation_id and (self.config.persistence or self.config.event_log):
                files = self.store._snapshot_session_files(self._state_root, self._conversation_id)
            self._ingest(self._align_host_observer_messages(list(messages or [])))
            self.store.protect_recent_turns(
                self.config.protect_latest_user_turns,
                self.config.protect_latest_assistant_turns)
            if settled_recovery is not None:
                self._retire_recovery_scaffolding()
            self._persist()
            self._publish_context_transaction()
        except BaseException as exc:
            self._rollback_direct_edit(snapshot, files)
            if not isinstance(exc, Exception):
                raise
            logger.warning("CLM on_turn_complete ingest failed", exc_info=True)
            if isinstance(exc, OSError) or (self._origin_mode == "native" and type(exc).__name__ == "SourceProjectionError"):
                raise

    # ------------------------------------------------------------------ #
    # select_context — the heart of the engine                            #
    # ------------------------------------------------------------------ #

    def _require_edit(self, resolved, kind):
        self.store.require_edit(resolved, kind)

    def _record_render_diagnostic(self, selected: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Record per-call prompt hash and first-divergence offset vs previous call."""
        import hashlib
        try:
            cur_bytes = json.dumps(selected, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
            prompt_hash = hashlib.sha256(cur_bytes).hexdigest()[:16]
            prev_msgs = self._last_rendered_messages
            prev_bytes = self._last_rendered_bytes
            cache_file = None
            if self._conversation_id and self.config.persistence:
                safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in self._conversation_id)
                cache_file = f"/tmp/hermes_clm_last_render_{safe_id}.json"
                if prev_bytes is None and os.path.exists(cache_file):
                    try:
                        with open(cache_file, "r", encoding="utf-8") as fh:
                            cached = json.load(fh)
                        if isinstance(cached, list):
                            prev_msgs = cached
                            prev_bytes = json.dumps(prev_msgs, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
                    except Exception:
                        pass
            div_byte: Optional[int] = None
            div_msg: Optional[int] = None
            div_ratio: Optional[float] = None
            if prev_bytes is not None and prev_msgs is not None:
                if prev_bytes != cur_bytes:
                    div_byte = next(
                        (i for i, (a, b) in enumerate(zip(prev_bytes, cur_bytes)) if a != b),
                        min(len(prev_bytes), len(cur_bytes)),
                    )
                    div_msg = next(
                        (i for i, (a, b) in enumerate(zip(prev_msgs, selected)) if a != b),
                        min(len(prev_msgs), len(selected)),
                    )
                    if len(prev_bytes) > 0:
                        div_ratio = round(div_byte / len(prev_bytes), 4)
            diag = {
                "prompt_hash": prompt_hash,
                "wire_bytes": len(cur_bytes),
                "prev_wire_bytes": len(prev_bytes) if prev_bytes is not None else None,
                "message_count": len(selected),
                "prev_message_count": len(prev_msgs) if prev_msgs is not None else None,
                "first_divergence_offset": div_byte,
                "first_divergence_msg_index": div_msg,
                "divergence_ratio": div_ratio,
                "recovery_state": self._recovery_turn.state,
                "recovery_episode": self._recovery_turn.episode,
            }
            self._last_rendered_messages = deepcopy(selected)
            self._last_rendered_bytes = cur_bytes
            self._last_render_diagnostic = diag
            self._render_diagnostics = (self._render_diagnostics + [diag])[-64:]
            if cache_file:
                try:
                    with open(cache_file, "w", encoding="utf-8") as fh:
                        json.dump(selected, fh, ensure_ascii=False, separators=(",", ":"))
                except Exception:
                    pass
            logger.info(
                "CLM render: hash=%s bytes=%d msgs=%d div_byte=%s div_msg=%s div_ratio=%s recovery=%s",
                prompt_hash, len(cur_bytes), len(selected), div_byte, div_msg, div_ratio, self._recovery_turn.state,
            )
        except Exception:
            logger.debug("CLM render diagnostic failed", exc_info=True)
        return selected

    def select_context_with_projection(self, request_messages, *, conversation_messages,
                                       incoming_message=None, budget_tokens=0, source_projection):
        self._require_origin()
        if self._origin_mode != "native" or self.store.coordinate_origin is None:
            projection_refuse("coordinate_resume_unproved", "", "Native lifecycle must precede projected selection.")
        represented = validate_source_projection(source_projection, conversation_messages,
                                                request_messages, incoming_message, self.store.messages)
        snapshot = self._begin_transaction()
        try:
            self._ingest(conversation_messages)
            self.store._projection_policy = {source_projection["sources"][i]["uid"]: deepcopy(row["policy"])
                                             for i, row in represented.items()}
            self.store._projection_context = deepcopy({"canonical": conversation_messages,
                "wire": request_messages, "incoming": incoming_message, "projection": source_projection,
                "narratives": self.store.narrative_memberships()})
            self._refresh_exact_pressure_notice()
            selected = self._attach_recovery_tail_echo(self._attach_context_map(
                self._attach_exact_pressure_notice(
                    self._attach_clm_guidance(self.selector.build_projected(request_messages, source_projection)))))
            if budget_tokens:
                self.budget.provider_context_length = max(self.budget.provider_context_length, int(budget_tokens))
            self._last_working_tokens = self.store.working_set_tokens()
            non_system = [m for m in request_messages if m.get("role") not in {"system", "developer"}]
            self._pending_estimate = {"tokens": self.counter.count_messages(non_system),
                                      "chars": sum(_content_chars(m) for m in non_system)}
            self._publish_context_transaction()
            return self._record_render_diagnostic(selected)
        except BaseException:
            self._rollback_transaction(snapshot)
            raise

    def select_context(self, request_messages: List[Dict[str, Any]], *,
                       conversation_messages: List[Dict[str, Any]] = None,
                       incoming_message: Dict[str, Any] = None,
                       budget_tokens: int = 0) -> Optional[List[Dict[str, Any]]]:
        self._require_origin()
        if self.store.coordinate_origin is not None:
            projection_refuse("projection_missing", "", "Marked native history requires projected selection.")
        self.config.validate_retention()
        if not isinstance(request_messages, list) or not request_messages:
            return None
        validation = validate_tool_protocol(request_messages)
        if not validation.valid:
            raise ProtocolError(validation)
        try:
            if not self._conversation_id:  # defensive: session start missed
                self.on_session_start(self._session_id or "default")
            self.store.validate_accepted()
            if budget_tokens:
                self.budget.provider_context_length = max(
                    self.budget.provider_context_length, int(budget_tokens))

            # Legacy standalone callers used a singleton *new* user as an
            # incoming-only request after on_turn_complete. Retain this narrow
            # compatibility path; host calls supply conversation/incoming context
            # and always use their full request as the authoritative history.
            ingest_messages = list(request_messages)
            if (conversation_messages is None and incoming_message is None
                    and len(ingest_messages) == 1 and ingest_messages[0].get("role") == "user"
                    and self.store.order
                    and envelope_digest(provider_envelope(ingest_messages[0])) != envelope_digest(
                        provider_envelope(self.store.messages[self.store.order[0]].envelope))):
                ingest_messages = [self.store.messages[u].envelope for u in self.store.order] + ingest_messages
            elif self._origin_mode != "native" and (conversation_messages is not None or incoming_message is not None):
                ingest_messages = self._align_host_request_messages(ingest_messages)
            self._ingest(ingest_messages)
            if self.config.budget_mode == "exact":
                # Estimates remain telemetry; only the final request guard can
                # authorize an exact-mode send. Never trim sources heuristically.
                self._refresh_exact_pressure_notice()
                return self._render_budget_messages(self.store)
            working = self.store.working_set_tokens()
            # Bounded recovery turn (production host path): arm from the measured
            # crossing / retire the episode once pressure clears.
            self._observe_host_pressure(working)
            notice: Optional[str] = None

            if self._overflow:
                if working < self.budget.hard_limit:
                    # Model compacted its way out: overflow resolved.
                    self._overflow = False
                    self._overflow_retries = 0
                    self.budget.reset_overflow_retries()
                    notice = ("CLM overflow resolved: working context is back "
                              "within budget. Continue the task.")
                elif self._overflow_retries > self.config.overflow_compaction_retries:
                    # Retries exhausted: host emergency fallback (spec §18/19).
                    self._emergency_compress(initiator="host-emergency")
                    working = self.store.working_set_tokens()
                    notice = (
                        "CLM EMERGENCY COMPACTION (host): model-directed "
                        "compaction did not free space in time. Deterministic "
                        "removal of summarized spans and stale tool output "
                        "applied; protected messages and recent turns preserved. "
                        "Continue the task.")
                else:
                    self._overflow_retries += 1
                    retries_left = (self.config.overflow_compaction_retries
                                    - self._overflow_retries + 1)
                    self.telemetry.record("overflow", initiator="host",
                                          active_tokens=working,
                                          retries_left=retries_left)
                    notice = self.selector.overflow_notice(retries_left)
            elif working > self.budget.hard_limit:
                self._overflow = True
                self._overflow_retries = 0
                self.telemetry.record("overflow", initiator="host",
                                      active_tokens=working,
                                      retries_left=self.config.overflow_compaction_retries)
                notice = self.selector.overflow_notice(
                    self.config.overflow_compaction_retries)
            else:
                level = self.budget.nudge_level(working)
                if level is not None and self.config.emit_nudges:
                    self.telemetry.record("nudge", initiator="host",
                                          level=level, active_tokens=working)
                    notice = self.selector.notice_for_level(level)

            if self._recovery_turn.state == 'offered':
                # Keep the prompt-top prefix byte-stable across episode arming and
                # intra-turn tool calls: on the approximate host path, the bounded
                # recovery directive is delivered exclusively at the tail via
                # _attach_recovery_tail_echo, and physical history rewrites take
                # effect at the next turn boundary (on_turn_complete).
                notice = None
                turn = self._recovery_turn
                if turn.frozen_prefix is not None and turn.frozen_order_len is not None:
                    selected = deepcopy(turn.frozen_prefix)
                    for uid in self.store.order[turn.frozen_order_len:]:
                        rec = self.store.messages.get(uid)
                        if rec is not None and rec.active:
                            rendered_rec = self.selector._render_record(rec)
                            if rendered_rec is not None:
                                selected.append(rendered_rec)
                    self._last_working_tokens = working
                    non_system = [m for m in request_messages
                                  if isinstance(m, dict) and m.get("role") != "system"]
                    self._pending_estimate = {
                        "tokens": self.counter.count_messages(non_system),
                        "chars": sum(_content_chars(m) for m in non_system),
                    }
                    return self._record_render_diagnostic(self._attach_recovery_tail_echo(selected))
            elif self._recovery_turn_enabled() and not self.config.emit_nudges:
                notice = None
            selected = self.selector.build(notice_text=notice)
            selected = self._attach_clm_guidance(selected)
            if (self._overflow or working > self.budget.hard_limit) and self._recovery_turn.state != 'offered':
                # Spec §18 step 2: reject the pending expansion — the selected
                # request itself must fit the hard limit. Trim oldest
                # non-protected content (presentation-only; the canonical
                # stream is untouched and the model is nudged to compact).
                selected, trimmed_tokens = self._trim_to_fit(selected)
                if trimmed_tokens > 0:
                    self.telemetry.record("trim", initiator="host",
                                          active_tokens=working,
                                          trimmed_tokens=trimmed_tokens)

            self._last_working_tokens = working
            non_system = [m for m in request_messages
                          if isinstance(m, dict) and m.get("role") != "system"]
            self._pending_estimate = {
                "tokens": self.counter.count_messages(non_system),
                "chars": sum(_content_chars(m) for m in non_system),
            }
            mapped = self._attach_context_map(selected)
            if self._recovery_turn.state == 'offered' and self._recovery_turn.frozen_prefix is None:
                self._recovery_turn.frozen_prefix = deepcopy(mapped)
                self._recovery_turn.frozen_order_len = len(self.store.order)
            return self._record_render_diagnostic(self._attach_recovery_tail_echo(mapped))
        except ValueError:
            # Synchronous local rejection. The host still catches selector errors;
            # Task 3 must gate its fail-open fallback before transport.
            raise
        except Exception:
            # Fail-open (host contract): never break the request over engine bugs.
            logger.warning("CLM select_context failed; returning original request",
                           exc_info=True)
            return None

    # ------------------------------------------------------------------ #
    # tools (spec §9)                                                     #
    # ------------------------------------------------------------------ #

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        from .tools import tool_schemas
        return tool_schemas(context_map_enabled=self.config.context_map_enabled)

    @observe_reference_restore
    def handle_tool_call(self, name: str, args: Dict[str, Any], **kwargs: Any) -> str:
        from .tools import dispatch
        if self.store._origin_admission != "admitted":
            return json.dumps({"ok": False, "error": "coordinate_resume_unproved",
                               "projection_error": deepcopy(self.store._origin_reason)})
        try:
            if self.config.budget_mode == 'exact' and name in _MUTATING_TOOLS and 'messages' in kwargs:
                result = self._dispatch_budgeted_tool(name, args, kwargs['messages'], kwargs.get('tool_call_id'))
            else:
                result = dispatch(self, name, args or {}, **kwargs)
        except PolicyViolation as exc:
            result = json.dumps({"ok": False, "error": exc.code,
                                 "message": exc.message,
                                 "message_ids": exc.message_ids,
                                 "required_tokens": exc.required_tokens,
                                 "available_tokens": exc.available_tokens})
        except Exception as exc:
            logger.warning("CLM tool %s failed: %s", name, exc)
            result = json.dumps({"ok": False, "error": "internal_error",
                                 "message": str(exc)})
        # Bounded recovery turn: record the observed operation for settlement.
        self._observe_recovery_tool(name, result)
        return result

    # ------------------------------------------------------------------ #
    # status (host + CLI)                                                 #
    # ------------------------------------------------------------------ #

    def get_status(self) -> Dict[str, Any]:
        if self.store._origin_admission != "admitted":
            return {"ok": False, "error": "coordinate_resume_unproved",
                    "projection_error": deepcopy(self.store._origin_reason)}
        working = self.store.working_set_tokens()
        if _HostBase is not None:
            base = super().get_status()
        else:
            base = {
                "last_prompt_tokens": max(self.last_prompt_tokens, 0),
                "threshold_tokens": self.threshold_tokens,
                "context_length": self.context_length,
                "usage_percent": min(100, working / self.budget.usable_tokens * 100)
                if self.budget.usable_tokens else 0,
                "compression_count": self.compression_count,
            }
        base.update(self.budget.status(self.store, working))
        base.update(self.budget.metrics_snapshot())
        base["working_set_tokens"] = working
        base["metrics"] = self.telemetry.snapshot()
        base['reference_restore_observations'] = self.telemetry.reference_restore_snapshot()
        base["backend"] = self.counter.backend_name
        base["state_root"] = self._state_root
        base["conversation_id"] = self._conversation_id
        base["uid_version"] = 2
        base["narrative_memberships"] = self.store.narrative_memberships()
        base["recovery_required"] = self.store.recovery_required
        base.update(self.exact_budget_status())
        if self._recovery_turn_enabled():
            base["recovery_turn"] = self._recovery_turn.status()
        last = self.store.metrics.get("last_compaction_at")
        if last:
            base["last_compaction_at"] = last
        return base

    def status_text(self) -> str:
        s = self.get_status()
        if s.get("ok") is False:
            return json.dumps(s)
        lines = [
            f"Conversation: {s.get('conversation_id') or '-'}",
            "Context engine: CLM",
            "",
            f"Working budget:     {s['budget_tokens']:,}",
            f"Active tokens:      {s['working_set_tokens']:,}",
            f"Utilization:        {round(s['utilization'] * 100, 1)}%",
            f"Archived messages:  {s['archived_messages']}",
            f"Synthetic summaries: {s['summaries']}",
            f"Context revision:   {s['context_revision']}",
            f"Provider window:    {s['provider_context_length']:,}",
            f"Tokenizer backend:  {s['backend']}",
        ]
        if s.get("last_compaction_at"):
            lines.append(f"Last compaction:    {s['last_compaction_at']}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # internals                                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _uid_of(message: Dict[str, Any], index: int) -> str:
        return envelope_uid(index, message)

    @staticmethod
    def _normalize_host_wire_message(message: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(message, dict):
            return message
        m = deepcopy(message)
        api_content = m.pop("api_content", None)
        if isinstance(api_content, str) and api_content and m.get("role") in ("user", "assistant"):
            m["content"] = api_content
        m.pop("finish_reason", None)
        m.pop("_length_continuation_fragment", None)
        m.pop("_length_continuation_nudge", None)
        if isinstance(m.get("tool_calls"), list):
            m["tool_calls"] = [
                {k: v for k, v in tc.items() if k not in ("call_id", "response_item_id")}
                if isinstance(tc, dict) else tc
                for tc in m["tool_calls"]
            ]
        return provider_envelope(m)

    @staticmethod
    def _host_core_envelope(message: Dict[str, Any]) -> Dict[str, Any]:
        """Strip host SQLite reload transient keys (tool_name/name/_db_persisted) for slot matching."""
        transient = {
            "name", "tool_name", "_db_persisted", "finish_reason", "api_content",
            "_length_continuation_fragment", "_length_continuation_nudge",
        }
        env = provider_envelope(message)
        return {k: v for k, v in env.items() if k not in transient}

    def _align_host_request_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Preserve existing canonical envelopes across SQLite --oneshot reloads when only
        host persistence metadata (e.g. tool 'name' vs 'tool_name' or '_db_persisted') drifted."""
        if self._origin_mode == "native" or not messages or not self.store.order:
            return messages
        aligned: List[Dict[str, Any]] = []
        for idx, raw in enumerate(messages):
            if isinstance(raw, dict) and idx < len(self.store.order):
                existing = self.store.messages[self.store.order[idx]].envelope
                norm = self._normalize_host_wire_message(raw)
                if (existing.get("role") == norm.get("role")
                        and existing.get("content") == norm.get("content")
                        and existing.get("tool_call_id") == norm.get("tool_call_id")
                        and existing.get("tool_calls") == norm.get("tool_calls")
                        and envelope_digest(self._host_core_envelope(existing)) == envelope_digest(self._host_core_envelope(norm))):
                    aligned.append(deepcopy(existing))
                    continue
            aligned.append(raw)
        return aligned

    def _align_host_observer_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Align host observer transcripts (which omit leading system/developer
        messages and retain durable fields like finish_reason / call_id) with
        the active wire prefix ingested by select_context."""
        if (self._origin_mode == "native" or not messages or not self.store.order
                or not isinstance(messages[0], dict)
                or messages[0].get("role") in {"system", "developer"}):
            return messages
        prefix_uids: List[str] = []
        for uid in self.store.order:
            rec = self.store.messages.get(uid)
            if rec is not None and rec.role in {"system", "developer"}:
                prefix_uids.append(uid)
            else:
                break
        if not prefix_uids or len(self.store.order) <= len(prefix_uids):
            return messages
        first_non_sys = self.store.messages[self.store.order[len(prefix_uids)]].envelope
        first_norm = self._normalize_host_wire_message(messages[0])
        if envelope_digest(provider_envelope(first_norm)) != envelope_digest(provider_envelope(first_non_sys)):
            return messages
        aligned = [deepcopy(self.store.messages[u].envelope) for u in prefix_uids]
        offset = len(prefix_uids)
        for idx, raw in enumerate(messages):
            norm = self._normalize_host_wire_message(raw)
            if offset + idx < len(self.store.order):
                existing = self.store.messages[self.store.order[offset + idx]].envelope
                if (existing.get("role") == norm.get("role")
                        and existing.get("content") == norm.get("content")
                        and existing.get("tool_call_id") == norm.get("tool_call_id")
                        and existing.get("tool_calls") == norm.get("tool_calls")
                        and envelope_digest(self._host_core_envelope(existing)) == envelope_digest(self._host_core_envelope(norm))):
                    aligned.append(deepcopy(existing))
                    continue
            aligned.append(norm)
        return aligned

    def _ingest(self, messages: List[Dict[str, Any]]) -> None:
        """Idempotent ingest of the current request (incl. the system slot)."""
        self._require_origin()
        snapshot = self.store._snapshot()
        counter = deepcopy(vars(self.counter))
        try:
            if self._origin_mode == "native":
                validate_json(messages)
                validate_source_conflicts(messages, self.store.messages)
                self.store._canonical_ingestion = True
            self.store.ingest_messages(list(messages))
            self.store._canonical_ingestion = False
            self.store.validate_accepted()
        except BaseException:
            self.store._restore_snapshot(snapshot)
            self.counter.__dict__ = counter
            raise
        self.budget.observe(self.store.working_set_tokens())
        # Canonical tails/rewinds are reconciled by the store. Offloading does
        # not replace provider bodies implicitly; explicit body compaction owns
        # that request change and its provenance.

    def _transaction_snapshot(self) -> dict:
        # Detached inspection remains supported between admission and the host's
        # acknowledgment. Mutators still enter through normal origin admission.
        self.store.require_origin(inspection=True)
        return self._capture_transaction_snapshot()

    def _capture_transaction_snapshot(self) -> dict:
        runtime = deepcopy({k: v for k, v in vars(self).items()
                            if k not in {"config", "store", "counter", "budget", "selector", "telemetry", "_exact_counter", "_request_renderer", "_pending_tool_budget", "_pending_direct_rollback", "_pending_tool_publication", "_tool_budget_lock", "_last_rendered_messages", "_last_rendered_bytes", "_last_render_diagnostic", "_render_diagnostics"}})
        budget = deepcopy({k: v for k, v in vars(self.budget).items() if k != "config"})
        counter = deepcopy(vars(self.counter))
        return {"store": self.store._snapshot(), "runtime": runtime,
                "budget": budget, "counter": counter}

    def _begin_transaction(self) -> dict:
        snapshot = self._transaction_snapshot()
        self.store.begin_transaction()
        return snapshot

    def _rollback_transaction(self, snapshot: dict) -> None:
        self.store._restore_snapshot(snapshot["store"])
        services = {"config", "store", "counter", "budget", "selector", "telemetry", "_exact_counter", "_request_renderer", "_pending_tool_budget", "_pending_direct_rollback", "_pending_tool_publication", "_tool_budget_lock", "_last_rendered_messages", "_last_rendered_bytes", "_last_render_diagnostic", "_render_diagnostics"}
        for k in list(vars(self)):
            if k not in services:
                del self.__dict__[k]
        self.__dict__.update(deepcopy(snapshot["runtime"]))
        self.budget.__dict__ = {"config": self.budget.config, **deepcopy(snapshot["budget"])}
        self.counter.__dict__ = deepcopy(snapshot["counter"])

    def _attach_clm_guidance(self, selected: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Append the CLM contract to the system prefix (spec §15)."""
        prefix = 0
        while prefix < len(selected) and selected[prefix].get("role") in {"system", "developer"}:
            prefix += 1
        # A distinct instruction preserves all original content values/types.
        if prefix:
            selected.insert(prefix, {"role": "system", "content": CLM_SYSTEM_GUIDANCE})
        return selected

    # ------------------------------------------------------------------ #
    # overflow trim (spec §18 step 2)                                     #
    # ------------------------------------------------------------------ #

    def _trim_to_fit(self, selected: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], int]:
        """Shrink a selected request to the hard limit.

        Presentation-only safety valve: drops the OLDEST non-protected content
        from this request. The leading system/notice prefix and everything from
        the most recent user turn onward (current work + tool results) are kept.
        Nothing is archived or destroyed — the canonical stream is untouched,
        and the overflow notice is nudging the model to compact properly.
        """
        if not selected:
            return selected, 0
        validation = validate_tool_protocol(selected)
        if not validation.valid:
            raise ProtocolError(validation)
        limit = self.budget.hard_limit
        total_before = self.counter.count_messages(selected)
        total = total_before
        keep = set(range(len(selected)))
        units = []
        grouped = set()
        for group in validation.groups:
            grouped.update(group.source_indices)
            units.append(set(group.source_indices))
        units.extend({i} for i in range(len(selected)) if i not in grouped)
        units.sort(key=min)
        # Presentation trimming respects instructions, root, recent windows and
        # explicit pins. Store protection is recomputed at ingestion boundaries.
        users = [i for i, m in enumerate(selected) if m.get("role") == "user"]
        assistants = [i for i, m in enumerate(selected) if m.get("role") == "assistant"]
        protected = {i for i, m in enumerate(selected) if m.get("role") in {"system", "developer"}}
        if users:
            protected.update({users[0], users[-1]})
        if self.config.protect_latest_user_turns:
            protected.update(users[-self.config.protect_latest_user_turns:])
        if self.config.protect_latest_assistant_turns:
            protected.update(assistants[-self.config.protect_latest_assistant_turns:])
        # UIDs aren't on wire views. Compare complete envelopes to locate pins.
        pins = {envelope_digest(provider_envelope(rec.envelope)) for rec in self.store.messages.values()
                if "pinned" in rec.permanent_reasons or rec.metadata.get("clm_pin")}
        protected.update(i for i, m in enumerate(selected) if envelope_digest(m) in pins)
        for unit in units:
            if total <= limit:
                break
            if unit.intersection(protected):
                continue
            keep.difference_update(unit)
            total -= sum(self.counter.count_message(selected[i]) for i in unit)
        out = [provider_envelope(m) for i, m in enumerate(selected) if i in keep]
        if total > limit and self.config.budget_mode == "exact":
            raise PolicyViolation("protected_context_overflow",
                                  "protected complete groups exceed the working limit")
        return out, max(0, total_before - total)

    # ------------------------------------------------------------------ #
    # operations (atomic per spec §33)                                    #
    # ------------------------------------------------------------------ #

    @_current_protection
    def apply_lookup(self, query, max_results=_OMITTED_LIMIT, cursor=None,
                     initiator='model'):
        """Present metadata under the same owned publication as other tools.

        A direct exact call certifies its prospective source frame. Native
        dispatch additionally admits the actual prepared reply before commit.
        """
        import time
        from datetime import datetime
        from zoneinfo import ZoneInfo
        from .retrieval.context_map import _digest
        from .retrieval.lookup import plan_lookup, finish_lookup
        started = time.monotonic_ns()
        index = self.store.context_map_index
        plan = plan_lookup(index, query, max_results, cursor)
        snapshot = self._begin_transaction()
        before = snapshot['store']['_context_map']
        revision_before = self.store.revision
        try:
            index._record_lookup_access(plan['node_ids'])
            if plan['node_ids']:
                self.store.bump_revision()
            result = finish_lookup(index, plan)
            lookup_id = 'cm_lookup_' + _digest([before['scope'], before['revision'],
                self.store.metrics.get('context_lookups', 0) + 1, query, cursor])[:32]
            result['lookup_id'] = lookup_id
            candidates = [dict(rank=r['rank'], node_id=r['host']['node_id'],
                archive_ref=r['host']['archive_ref'], current_branch=r['host']['current_branch'],
                score=r['score'], score_components=deepcopy(r['score_components'])) for r in result['results']]
            self._lookup_receipts = (self._lookup_receipts + [dict(scope=deepcopy(before['scope']),
                lookup_id=lookup_id, query=query, normalized_query=plan['normalized_query'],
                candidates=deepcopy(candidates))])[-32:]
            completed = time.monotonic_ns()
            self.telemetry.record('lookup', initiator=initiator,
                revision_before=revision_before, revision_after=self.store.revision,
                lookup_id=lookup_id, query=query, normalized_query=plan['normalized_query'],
                input_cursor=cursor, next_cursor=result['next_cursor'], max_results=plan['max_results'],
                ranking_version=result['ranking_version'], rank_scope=result['rank_scope'],
                map_revision_before=before['revision'], map_revision_after=result['map_revision'],
                candidate_count=len(result['results']),
                candidates=candidates,
                node_access=[dict(node_id=key, before=before['nodes'][key]['access_count'],
                    after=before['nodes'][key]['access_count']+1) for key in plan['node_ids']],
                started_monotonic_ns=started, completed_monotonic_ns=completed,
                lookup_latency_ms=(completed-started)/1_000_000,
                timestamp_detroit=datetime.now(ZoneInfo('America/Detroit')).isoformat(),
                source_consultation=None, recall_at_k=None, wrong_restore=None, autonomous_success=None)
            self._persist()
            self._publish_context_transaction()
            return result
        except Exception:
            self._rollback_transaction(snapshot)
            raise

    def _map_event_fields(self, before, source_uids):
        if before is None:
            return {}
        after = self.store.context_map_index.to_dict()
        sources = sorted(set(source_uids), key=lambda uid: self.store.get(uid).index)
        introduced = set(after['presentations']) - set(before['presentations'])
        for uid, node in after['nodes'].items():
            introduced.update(set(node['presentation_refs'])
                              - set(before['nodes'].get(uid, {}).get('presentation_refs', [])))
        return {'context_map_delta': {
            'revision_before': before['revision'], 'revision_after': after['revision'],
            'node_ids': [uid for uid, node in after['nodes'].items() if node != before['nodes'].get(uid)],
            'archive_refs': [ref for ref, record in after['archive_refs'].items()
                             if set(record['source_uids']).issubset(sources)],
            'presentation_refs': [pid for pid in after['presentations'] if pid in introduced],
            'source_uids': sources,
        }}

    def _map_proposal(self, path, description, keywords):
        if all(value is _OMITTED_MAP_METADATA for value in (path, description, keywords)):
            return None
        if not self.config.context_map_enabled:
            raise PolicyViolation('context_map_disabled', 'Context Map authoring is disabled')
        if path is _OMITTED_MAP_METADATA:
            raise PolicyViolation('map_path_required', 'Map metadata requires map_path')
        from .retrieval.context_map import _text, _keywords
        description = '' if description is _OMITTED_MAP_METADATA else description
        keywords = [] if keywords is _OMITTED_MAP_METADATA else keywords
        try:
            if type(path) is not list or not 1 <= len(path) <= 5 or type(keywords) is not list:
                raise ValueError('invalid_context_map_metadata_shape')
            for title in path:
                _text(title, 96, empty=False)
            _text(description, 256)
            words = _keywords(keywords)
        except ValueError as exc:
            raise PolicyViolation('invalid_map_metadata', str(exc)) from exc
        return {'path': list(path), 'description': description, 'keywords': words}

    def _place_map_proposal(self, proposal, source_uids, before_presentations,
                            *, summary_id=None, body_uids=()):
        if proposal is None:
            return
        from .retrieval.context_map import ContextMapIndex, _digest
        index = self.store.context_map_index
        try:
            node = index.create_node(proposal['path'], source_uids,
                                     description=proposal['description'], keywords=proposal['keywords'])
        except ValueError as exc:
            # The facade owns normalized reuse and current creation limits.
            # Its refusal aborts the enclosing source-operation transaction.
            raise PolicyViolation('invalid_map_metadata', str(exc)) from exc
        data = index.to_dict()
        body_digests = {uid: _digest(self.store.get(uid).body_compaction)
                        for uid in body_uids if self.store.get(uid).body_compaction is not None}
        selected = [pid for pid, record in data['presentations'].items()
                    if (record['kind'] == 'summary' and record['owner_uid'] == summary_id)
                    or (record['kind'] == 'body' and record['owner_uid'] in body_digests
                        and record['payload_digest'] == body_digests[record['owner_uid']])]
        if not selected:
            return
        fresh = set(selected) - before_presentations
        for current in data['nodes'].values():
            # Only this operation's new presentations leave their automatic
            # placement. Earlier associations, including shared ones, survive.
            current['presentation_refs'] = [pid for pid in current['presentation_refs'] if pid not in fresh]
        target = data['nodes'][node.node_id]
        target['presentation_refs'].extend(pid for pid in selected if pid not in target['presentation_refs'])
        for current in data['nodes'].values():
            summaries = [data['presentations'][pid]['owner_uid'] for pid in current['presentation_refs']
                         if data['presentations'][pid]['kind'] == 'summary']
            current['summary_refs'] = [uid for uid in current['summary_refs'] if uid in summaries]
            for uid in summaries:
                if uid not in current['summary_refs']:
                    current['summary_refs'].append(uid)
        ContextMapIndex.from_dict(data, store=self.store)

    @_current_protection
    def apply_compact(self, message_ids: List[str], summary: str,
                      reason: str = "", initiator: str = "model",
                      form: str = "summary", *, map_path=_OMITTED_MAP_METADATA,
                      map_description=_OMITTED_MAP_METADATA,
                      keywords=_OMITTED_MAP_METADATA) -> Dict[str, Any]:
        self._require_origin()
        proposal = self._map_proposal(map_path, map_description, keywords)
        if form not in {"summary", "body"}:
            raise PolicyViolation("invalid_compaction_form", "use summary or body compaction")
        summary = (summary or "").strip()
        if not summary:
            raise PolicyViolation("empty_summary",
                                  "compact requires non-empty summary text")
        if len(summary) > 16000:
            raise PolicyViolation("summary_too_long",
                                  "summary exceeds 16,000 characters; split the span")
        if (self._recovery_turn_enabled() and self._recovery_turn.state == 'offered'
                and self.config.budget_mode == 'approximate' and self._origin_mode != 'native'
                and message_ids):
            valid_subset = [
                u for u in message_ids
                if u in self.store.order and (rec := self.store.get(u)) is not None
                and rec.active and not rec.protected
            ]
            if valid_subset:
                message_ids = valid_subset
        from .policies import validate_operation
        resolved, _ = validate_operation(
            self.store, self.config, op="compact", message_ids=message_ids,
            budget_controller=self.budget)
        self._require_edit(resolved, "body" if form == "body" else "narrative")
        snapshot = self._begin_transaction()
        revision_before = self.store.revision
        try:
            map_before = snapshot['store']['_context_map']
            before_presentations = (set(self.store.context_map_index.to_dict()['presentations'])
                                    if proposal is not None else set())
            tokens_before = self.store.working_set_tokens()
            source_tokens = sum((self.store.get(u).token_count or 0) for u in resolved)
            if form == "body":
                self.store.compact_bodies(resolved, summary, initiator)
                sources = resolved
                if map_before is not None:
                    sources = {uid for target in resolved
                               if self.store.get(target).body_compaction is not None
                               for uid in self.store.get(target).body_compaction['derivation_source_uids']}
                    sources = sorted(sources, key=lambda uid: self.store.get(uid).index)
                self._place_map_proposal(proposal, sources, before_presentations, body_uids=resolved)
                self.store.bump_revision()
                tokens_after = self.store.working_set_tokens()
                self.telemetry.record("compact_body", initiator=initiator,
                                      source_message_ids=resolved, reason=reason,
                                      revision_before=revision_before, revision_after=self.store.revision,
                                      tokens_before=tokens_before, tokens_after=tokens_after,
                                      **self._map_event_fields(map_before, sources))
                self._persist()
                self._publish_context_transaction()
                return {"ok": True, "message_ids": resolved, "form": "body",
                        "tokens_before": tokens_before, "tokens_after": tokens_after,
                        "tokens_saved": max(0, tokens_before-tokens_after),
                        "context_revision": self.store.revision}
            derivation = invariants.derive_inputs(invariants.capture(self.store), resolved)
            summary_rec = self.store.create_summary(summary, resolved,
                                                    created_by=initiator,
                                                    derivation_source_uids=derivation)
            self.store.mark_archived(resolved, narrative_summary_id=summary_rec.uid)
            for uid in resolved:
                rec = self.store.get(uid)
                if rec is not None and summary_rec.uid not in rec.summary_parent_ids:
                    rec.summary_parent_ids.append(summary_rec.uid)
            self._place_map_proposal(proposal, summary_rec.derivation_source_uids,
                                     before_presentations, summary_id=summary_rec.uid)
            if self._recovery_turn_enabled() and self._recovery_turn.state == 'offered':
                self._retire_recovery_scaffolding()
            self.store.bump_revision()
            tokens_after = self.store.working_set_tokens()
            saved = max(0, max(source_tokens - (summary_rec.token_count or 0), tokens_before - tokens_after))
            self.store.metrics["last_compaction_at"] = utc_now_iso()
            self.store.metrics["last_compaction_tokens"] = tokens_after
            self.telemetry.record(
                "compact", initiator=initiator,
                revision_before=revision_before, revision_after=self.store.revision,
                source_message_ids=resolved, summary_id=summary_rec.uid,
                derivation_source_uids=summary_rec.derivation_source_uids,
                tokens_before=tokens_before, tokens_after=tokens_after,
                tokens_saved=saved, summary_tokens=summary_rec.token_count or 0,
                reason=reason,
                **self._map_event_fields(map_before, summary_rec.derivation_source_uids),
            )
            self._persist()
            self._publish_context_transaction()
            return {
                "ok": True,
                "tokens_before": tokens_before,
                "tokens_after": tokens_after,
                "tokens_saved": saved,
                "summary_id": summary_rec.uid,
                "message_ids": resolved,
                "derivation_source_uids": deepcopy(summary_rec.derivation_source_uids),
                "form": "summary",
                "context_revision": self.store.revision,
            }
        except Exception:
            self._rollback_transaction(snapshot)
            raise

    @_current_protection
    def apply_archive(self, message_ids: List[str], reason: str = "",
                      initiator: str = "model", *, map_path=_OMITTED_MAP_METADATA,
                      map_description=_OMITTED_MAP_METADATA,
                      keywords=_OMITTED_MAP_METADATA) -> Dict[str, Any]:
        self._require_origin()
        proposal = self._map_proposal(map_path, map_description, keywords)
        if (self._recovery_turn_enabled() and self._recovery_turn.state == 'offered'
                and self.config.budget_mode == 'approximate' and self._origin_mode != 'native'
                and message_ids):
            valid_subset = [
                u for u in message_ids
                if u in self.store.order and (rec := self.store.get(u)) is not None
                and rec.active and not rec.protected
            ]
            if valid_subset:
                message_ids = valid_subset
        from .policies import validate_operation
        resolved, _ = validate_operation(
            self.store, self.config, op="archive", message_ids=message_ids,
            budget_controller=self.budget)
        self._require_edit(resolved, "archive")
        snapshot = self._begin_transaction()
        revision_before = self.store.revision
        try:
            tokens_before = self.store.working_set_tokens()
            removed = sum((self.store.get(u).token_count or 0) for u in resolved)
            self.store.mark_archived(resolved)
            self._place_map_proposal(proposal, resolved, set())
            if self._recovery_turn_enabled() and self._recovery_turn.state == 'offered':
                self._retire_recovery_scaffolding()
            self.store.bump_revision()
            tokens_after = self.store.working_set_tokens()
            self.telemetry.record(
                "archive", initiator=initiator,
                revision_before=revision_before, revision_after=self.store.revision,
                source_message_ids=resolved, tokens_removed=removed,
                tokens_before=tokens_before, tokens_after=tokens_after,
                reason=reason,
                **self._map_event_fields(snapshot['store']['_context_map'], resolved),
            )
            self._persist()
            self._publish_context_transaction()
            return {
                "ok": True,
                "tokens_before": tokens_before,
                "tokens_after": tokens_after,
                "tokens_removed": removed,
                "archived_ids": resolved,
                "context_revision": self.store.revision,
            }
        except Exception:
            self._rollback_transaction(snapshot)
            raise

    @observe_reference_restore
    @_current_protection
    def apply_restore(self, message_ids: Optional[List[str]] = None, summary_id: str = "",
                      restore_sources: bool = True,
                      initiator: str = "model", return_plan: Optional[str] = None, *,
                      archive_refs=_OMITTED_RESTORE_ARGUMENT,
                      reason=_OMITTED_RESTORE_ARGUMENT) -> Dict[str, Any]:
        started_monotonic_ns = time.monotonic_ns()
        self._require_origin()
        if return_plan is not None and (type(return_plan) is not str
                or not return_plan.strip() or len(return_plan) > 2000):
            raise PolicyViolation('invalid_return_plan', 'Return plan must be nonempty text of at most 2000 characters')
        from .policies import validate_operation
        targets: List[str] = []
        reference_fields = None
        if archive_refs is not _OMITTED_RESTORE_ARGUMENT:
            if message_ids or summary_id:
                raise PolicyViolation('invalid_restore_arguments', 'Archive references cannot be combined with legacy targets')
            targets, reference_fields = plan_reference_restore(self, archive_refs, reason)
        elif reason is not _OMITTED_RESTORE_ARGUMENT:
            raise PolicyViolation('invalid_restore_arguments', 'Reference reason requires archive_refs')
        if summary_id:
            summary = self.store.summary_by_prefix(summary_id)
            if summary is None:
                raise PolicyViolation("unknown_summary",
                                      f"no such summary: {summary_id}")
            if restore_sources:
                targets.extend(summary.source_message_ids)
        if message_ids:
            targets.extend(message_ids)
        if not targets:
            raise PolicyViolation(
                "empty_operation",
                "nothing to restore (summary has no sources and no message_ids)")
        resolved, source_delta = validate_operation(
            self.store, self.config, op="restore", message_ids=targets,
            budget_controller=self.budget)
        snapshot = self._begin_transaction()
        revision_before = self.store.revision
        try:
            if return_plan is not None and self.config.budget_mode == 'exact':
                self.begin_budget_excursion('restore', return_plan)
            tokens_before = self.store.working_set_tokens()
            changed = [u for u in resolved if self.store.get(u).archived or self.store.get(u).body_compaction]
            restored_tokens = (sum((self.store.get(u).token_count or 0) for u in resolved)
                               if reference_fields is None else source_delta)
            if reference_fields is None or changed:
                self.store.mark_active(resolved)
            if summary_id:
                s = self.store.summary_by_prefix(summary_id)
                if s is not None:
                    s.active = False  # raw content takes its place
            if reference_fields is None or changed:
                self.store.bump_revision()
            tokens_after = self.store.working_set_tokens()
            extra = {}
            if reference_fields is not None:
                extra.update(reference_fields, changed_source_ids=changed, noop=not changed,
                    tokens_restored_scope='source_delta_estimate',
                    source_consultation=None, wrong_restore=None, autonomous_success=None,
                    **self._map_event_fields(snapshot['store']['_context_map'], resolved))
                completed_monotonic_ns = time.monotonic_ns()
                extra.update(started_monotonic_ns=started_monotonic_ns,
                    completed_monotonic_ns=completed_monotonic_ns,
                    restore_latency_ms=(completed_monotonic_ns - started_monotonic_ns) / 1_000_000,
                    restore_latency_scope='source_operation_before_publication',
                    publication_latency_ms=None, native_admission_latency_ms=None,
                    **restore_timestamps())
            self.telemetry.record(
                "restore", initiator=initiator,
                revision_before=revision_before, revision_after=self.store.revision,
                source_message_ids=resolved, summary_id=summary_id or None,
                tokens_restored=restored_tokens,
                tokens_before=tokens_before, tokens_after=tokens_after,
                **extra,
            )
            self._persist()
            self._publish_context_transaction()
            result = {
                "ok": True,
                "tokens_before": tokens_before,
                "tokens_after": tokens_after,
                "tokens_restored": restored_tokens,
                "restored_ids": resolved,
                "context_revision": self.store.revision,
            }
            if reference_fields is not None:
                nodes = reference_fields['source_node_ids']
                result.update(archive_refs=reference_fields['archive_refs'], source_node_ids=nodes[:5],
                    omitted_source_nodes=max(0, len(nodes)-5), changed_source_ids=changed,
                    noop=not changed, tokens_restored_scope='source_delta_estimate',
                    lookup_ids=reference_fields['lookup_ids'],
                    unattributed_archive_refs=reference_fields['unattributed_archive_refs'],
                    lookup_provenance_scope=reference_fields['lookup_provenance_scope'])
            return result
        except Exception:
            self._rollback_transaction(snapshot)
            raise

    @_current_protection
    def apply_replace_summary(self, summary_id: str, replacement: str,
                              initiator: str = "model") -> Dict[str, Any]:
        self._require_origin()
        replacement = (replacement or "").strip()
        if not replacement:
            raise PolicyViolation("empty_summary",
                                  "replace_summary requires non-empty replacement text")
        summary = self.store.summary_by_prefix(summary_id)
        if summary is None:
            raise PolicyViolation("unknown_summary",
                                  f"no such summary: {summary_id}")
        self._require_edit(summary.source_message_ids, "narrative")
        snapshot = self._begin_transaction()
        revision_before = self.store.revision
        try:
            tokens_before = self.store.working_set_tokens()
            updated = self.store.update_summary(summary.uid, replacement,
                                                created_by=initiator)
            self.store.bump_revision()
            tokens_after = self.store.working_set_tokens()
            self.telemetry.record(
                "replace_summary", initiator=initiator,
                revision_before=revision_before, revision_after=self.store.revision,
                summary_id=updated.uid,
                source_message_ids=updated.source_message_ids,
                derivation_source_uids=updated.derivation_source_uids,
                tokens_before=tokens_before, tokens_after=tokens_after,
                revision=updated.revision,
            )
            self._persist()
            self._publish_context_transaction()
            return {
                "ok": True,
                "summary_id": updated.uid,
                "revision": updated.revision,
                "tokens_before": tokens_before,
                "tokens_after": tokens_after,
                "context_revision": self.store.revision,
            }
        except Exception:
            self._rollback_transaction(snapshot)
            raise

    # ------------------------------------------------------------------ #
    # emergency host fallback (spec §19)                                  #
    # ------------------------------------------------------------------ #

    @_current_protection
    def _emergency_compress(self,
                            initiator: str = "host-emergency") -> Optional[Dict[str, Any]]:
        """Deterministic fallback order (spec §19):

        1. already-summarized raw spans (doubly redundant),
        2. old low-value tool outputs,
        3. oldest non-recent messages,
        until the working set is under the hard limit. Protected messages and
        recent turns are never touched. The canonical stream is never mutated.
        """
        self._require_origin()
        if not self.config.emergency_host_fallback:
            return None
        tokens_before = self.store.working_set_tokens()
        if tokens_before <= self.budget.hard_limit:
            return None
        snapshot = self._begin_transaction()
        revision_before = self.store.revision
        try:
            order = emergency_order(self.store, self.config)
            removed: List[str] = []
            headroom = tokens_before - self.budget.hard_limit
            for uid in order:
                if headroom <= 0:
                    break
                rec = self.store.get(uid)
                if rec is None or not rec.active or uid in removed:
                    continue
                try:
                    resolved = self.store.resolve_groups([uid])
                    # All members must be eligible; recent/tool closure is atomic.
                    if any(self.store.get(u).protected or u not in order for u in resolved):
                        continue
                    self.store.mark_archived(resolved)
                except PolicyViolation:
                    continue
                removed.extend(resolved)
                headroom -= sum(self.store.get(u).token_count or 0 for u in resolved)
            if removed:
                self.store.bump_revision()
            tokens_after = self.store.working_set_tokens()
            saved = max(0, tokens_before - tokens_after)
            self.telemetry.record(
                "emergency_compaction", initiator=initiator,
                revision_before=revision_before, revision_after=self.store.revision,
                source_message_ids=removed,
                tokens_before=tokens_before, tokens_after=tokens_after,
                tokens_removed=saved,
                **self._map_event_fields(snapshot['store']['_context_map'], removed),
            )
            self._persist()
            self._publish_context_transaction()
        except Exception:
            self._rollback_transaction(snapshot)
            raise
        # Engine/budget counters describe accepted compaction only. In
        # persistence-disabled mode _persist intentionally succeeds in memory.
        self.compression_count += 1
        self._overflow = False
        self._overflow_retries = 0
        self.budget.reset_overflow_retries()
        logger.warning(
            "CLM emergency compaction (%s): removed %d records, %s -> %s tokens",
            initiator, len(removed), tokens_before, tokens_after)
        return {"removed": removed, "tokens_before": tokens_before,
                "tokens_after": tokens_after}

    # ------------------------------------------------------------------ #
    # persistence                                                         #
    # ------------------------------------------------------------------ #

    def _persist(self) -> None:
        self._require_origin()
        if not self.store.recovery_required:
            self.store.validate_accepted()
        if self.store._budget_compound_depth:
            self._validate_store_budget(self.store, self._budget_operation)
        if (self._pending_tool_budget is not None
                and not self._pending_tool_budget.get('publishing')):
            return
        if (not self.config.persistence or not self._conversation_id
                or self.store.recovery_required):
            return
        path = self.store.save(self._state_root,
                               budget_tokens=self.budget.usable_tokens,
                               reserve_tokens=self.budget.reserve,
                               current_tokens=self.store.working_set_tokens())
        if path is None:
            raise OSError("CLM state was not saved")

    # ------------------------------------------------------------------ #
    # CLI helpers (spec §48/§49)                                          #
    # ------------------------------------------------------------------ #

    def list_active(self) -> List[Dict[str, Any]]:
        self._require_origin()
        return [
            {"uid": r.uid, "role": r.role, "index": r.index,
             "tokens": r.token_count, "protected": r.protected,
             "permanent_reasons": list(r.permanent_reasons), "temporary_reasons": list(r.temporary_reasons),
             "chars": len(r.content)}
            for r in self.store.active_messages()
        ]

    def list_archive(self) -> List[Dict[str, Any]]:
        self._require_origin()
        return [
            {"uid": r.uid, "role": r.role, "index": r.index,
             "tokens": r.token_count, "protected": r.protected,
             "permanent_reasons": list(r.permanent_reasons), "temporary_reasons": list(r.temporary_reasons),
             "chars": len(r.content)}
            for r in self.store.archived_messages()
        ]

    def show_summary(self, summary_id: str) -> Dict[str, Any]:
        self._require_origin()
        s = self.store.summary_by_prefix(summary_id)
        if s is None:
            return {"ok": False, "error": "unknown_summary"}
        return {"ok": True, **s.to_dict()}

    def recent_events(self, limit: int = 20) -> List[Dict[str, Any]]:
        self._require_origin()
        path = os.path.join(self._state_root, self._conversation_id, "events.jsonl")
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

    @_current_protection
    def reset_working_set(self) -> Dict[str, Any]:
        """Spec §49: reconstruct the active context from canonical history.

        Restore RAW originals on the current canonical branch. Historical
        records and summary provenance remain retained outside the view."""
        self._require_origin()
        self.store.validate_accepted()
        snapshot = self._begin_transaction()
        revision_before = self.store.revision
        try:
            self.store._projection_policy = {}
            if self.config.budget_mode != 'exact':
                self.store._projection_context = None
            for uid in self.store.order:
                rec = self.store.messages[uid]
                rec.archived = False
                rec.active = True
                self.store._retire_body_compaction(rec)
                rec.token_count = self.counter.count_message(rec.envelope)
            for s in self.store.summaries.values():
                s.active = False
            self.store._sync_context_map()
            self.store.recompute_protection()
            self.store.bump_revision()
            self.telemetry.record("reset_working_set", initiator="host",
                                  revision_before=revision_before,
                                  revision_after=self.store.revision)
            self._persist()
            self._publish_context_transaction()
            self._overflow = False
            self._overflow_retries = 0
            self.budget.reset_overflow_retries()
            return {"ok": True, "context_revision": self.store.revision,
                    "active_messages": len(self.store.active_messages())}
        except Exception:
            self._rollback_transaction(snapshot)
            raise
