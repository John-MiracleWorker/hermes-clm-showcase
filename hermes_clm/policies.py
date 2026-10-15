"""Policy / validation layer (spec §41, §32.6).

Enforces protected messages, recent-turn retention, maximum restoration size,
operation validity, and emergency-fallback eligibility. All validation is pure
(structural) — no model calls, no state mutation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .store import ContextStore


class PolicyViolation(Exception):
    """Raised when a context operation violates a policy rule.

    ``code`` is a stable machine-readable identifier for the event log.
    """

    def __init__(self, code: str, message: str, message_ids: Optional[List[str]] = None,
                 required_tokens: Optional[int] = None, available_tokens: Optional[int] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.message_ids = message_ids or []
        self.required_tokens = required_tokens
        self.available_tokens = available_tokens


def validate_operation(
    store: ContextStore,
    config: Any,
    *,
    op: str,
    message_ids: List[str],
    summary_id: str = "",
    restore_sources: bool = False,
    budget_controller: Any = None,
) -> Tuple[List[str], int]:
    """Validate a pending context operation (spec §32.6).

    Returns ``(resolved_message_ids, estimated_token_delta)`` or raises
    :class:`PolicyViolation`.
    """
    if op not in {"compact", "archive", "restore", "replace_summary"}:
        raise PolicyViolation("invalid_op", f"unknown operation: {op}")

    if op == "replace_summary":
        summary = store.summary_by_prefix(summary_id) if summary_id else None
        if summary is None:
            raise PolicyViolation("unknown_summary", f"no such summary: {summary_id}")
        delta = 0  # content length is re-estimated at apply time
        return [], delta

    if not message_ids:
        raise PolicyViolation("empty_operation", "no message ids supplied")

    message_ids = store.resolve_groups(message_ids)
    resolved: List[str] = []
    unknown: List[str] = []
    protected: List[str] = []
    for uid in message_ids:
        rec = store.get(uid)
        if rec is None:
            unknown.append(uid)
            continue
        if rec.protected:
            protected.append(uid)
            continue
        resolved.append(uid)

    if unknown:
        raise PolicyViolation("unknown_message_ids",
                              f"unknown message ids: {', '.join(unknown[:8])}", unknown)
    if protected:
        raise PolicyViolation(
            "protected_context",
            "operation references protected messages: " + ", ".join(protected[:8]),
            protected)
    if not resolved:
        raise PolicyViolation("empty_operation", "no eligible messages after validation")

    delta = 0
    if op == "restore":
        delta = _restore_delta(store, resolved, config, budget_controller)
    elif op in ("compact", "archive"):
        for uid in resolved:
            rec = store.get(uid)
            if rec is not None:
                delta -= (rec.token_count or 0)
        # compact: the summary content is counted at apply time; we estimate
        # here that the caller's summary is present (engine adds it after).
    return resolved, delta


def _restore_delta(store: ContextStore, uids: List[str], config: Any,
                   budget_controller: Any) -> int:
    added = 0
    for uid in uids:
        rec = store.get(uid)
        if rec is not None:
            original_tokens = store.counter.count_message(rec.envelope)
            if rec.archived:
                added += original_tokens
            elif rec.body_compaction:
                added += max(0, original_tokens - (rec.token_count or 0))
    cap = getattr(config, "max_restoration_tokens", 0)
    if cap and added > cap and config.budget_mode != 'exact':
        raise PolicyViolation(
            "max_restoration_exceeded",
            f"restoration of {added} tokens exceeds max_restoration_tokens={cap}")
    if budget_controller is not None and store is not None and config.budget_mode != 'exact':
        current = store.working_set_tokens()
        if budget_controller.will_overflow(current, added):
            raise PolicyViolation(
                "budget_exceeded",
                f"restoration needs {added} tokens; only "
                f"{max(0, budget_controller.hard_limit - current)} available",
                uids, required_tokens=added)
    return added


# --------------------------------------------------------------------------- #
# emergency host fallback (spec §19)                                          #
# --------------------------------------------------------------------------- #

def emergency_fallback_eligible(config: Any) -> bool:
    return bool(getattr(config, "emergency_host_fallback", True))


def emergency_order(store: ContextStore, config: Any) -> List[str]:
    """Deterministic emergency removal order (spec §19).

    1. remove already-summarized raw spans (doubly redundant)
    2. remove old low-value tool outputs
    3. preserve latest user/assistant work
    4. preserve unresolved decisions (heuristic: most recent assistant text
       mentions of open work — we keep all recent assistant turns)
    5. preserve all protected messages
    """
    protect_recent_u = max(config.protect_latest_user_turns, 2)
    protect_recent_a = max(config.protect_latest_assistant_turns, 1)

    users = [u for u in store.order if store.messages[u].role == "user"]
    assts = [u for u in store.order if store.messages[u].role == "assistant"]
    protected_recent = set(users[-protect_recent_u:] + assts[-protect_recent_a:])

    # Summarized uids: covered by an active summary.
    summarized: set = set()
    for s in store.summaries.values():
        if s.active:
            summarized.update(s.source_message_ids)

    def score(uid: str) -> Tuple[int, int]:
        rec = store.messages[uid]
        rank = 0
        if rec.protected or uid in protected_recent:
            rank = 9  # never removed
        elif uid in summarized:
            rank = 0  # remove first
        elif rec.role == "tool" or rec.metadata.get("tool_name"):
            rank = 1  # old tool outputs next
        elif rec.role == "assistant" and rec.index >= (store.messages[users[-1]].index if users else 0) - 4:
            rank = 2
        else:
            rank = 3
        return (rank, rec.index)  # within a rank, oldest first

    removable = [u for u in store.order
                 if not store.messages[u].protected and u not in protected_recent]
    removable.sort(key=score)
    return removable
