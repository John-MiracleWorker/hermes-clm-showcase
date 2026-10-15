"""Context selection (spec §7, §20).

Builds the effective working context for each provider request:

    SYSTEM / PROTECTED PREFIX
    CLM STATE NOTICE
    RELEVANT EARLIER SUMMARIES   (each at the anchor index of its span)
    ACTIVE CONVERSATION / WORKING SET
    MOST RECENT USER TURN
    CURRENT TOOL RESULTS

The canonical Hermes history is never mutated: ``select_context`` receives
cloned request messages and returns a fresh request-only list (host contract).
Chronological order is preserved; summaries render exactly at the logical
position of the removed span (spec §11 step 6) via a merged (index) timeline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from copy import deepcopy

from .budget import ContextBudgetController
from .prompts import build_notice, render_summary_message, urgent_overflow_notice
from .store import ContextStore
from .protocol import provider_envelope, validate_tool_protocol, ProtocolError, projection_refuse


def _clean(message: Dict[str, Any]) -> Dict[str, Any]:
    return provider_envelope(message)


class ContextSelector:
    """Stateless builder: pure function of (store, budget, notice)."""

    def __init__(self, store: ContextStore, budget: ContextBudgetController):
        self.store = store
        self.budget = budget

    # ------------------------------------------------------------------ #

    def build(self, *, notice_text: Optional[str] = None) -> List[Dict[str, Any]]:
        """Assemble the request-only message list (spec §20 order)."""
        out: List[Dict[str, Any]] = []

        self.store.require_envelopes()
        self.store.validate_accepted()
        # Instructions remain on the same chronological timeline as sources.
        # Synthetic notices follow the initial instruction prefix.
        # 3-6. Merged timeline: active messages and fully-archived summaries,
        # both keyed by canonical index (summaries render at their anchor).
        timeline: List[tuple] = []
        for rec in (self.store.messages[u] for u in self.store.order
                    if self.store.messages[u].active):
            timeline.append((rec.index, 0, rec))
        for s in self.store.active_summaries():
            if self.store.is_summary_fully_archived(s):
                timeline.append((s.anchor_index, 1, s))
        timeline.sort(key=lambda item: (item[0], item[1]))

        for _, kind, item in timeline:
            if kind == 0:
                message = self._render_record(item)
                if message is not None:
                    out.append(message)
            else:
                out.append(_clean(render_summary_message(item, item.anchor_index)))
        if notice_text:
            prefix = 0
            while prefix < len(out) and out[prefix].get("role") in {"system", "developer"}:
                prefix += 1
            out.insert(prefix, {"role": "system", "content": notice_text})
        validation = validate_tool_protocol(out)
        if not validation.valid:
            raise ProtocolError(validation)
        return out

    def _render_record(self, rec: Any) -> Optional[Dict[str, Any]]:
        envelope = rec.envelope
        if envelope is None:
            raise ValueError("missing canonical envelope")
        if rec.body_compaction:
            envelope["content"] = self.store.render_body(rec)
        return provider_envelope(envelope)

    def build_projected(self, request_messages, projection):
        """Render accepted choices onto today's detached, safety-adapted wire views."""
        self.store.validate_accepted()
        insertions = {}
        narrative_sources = set()
        for summary in self.store.active_summaries():
            if not self.store.is_summary_fully_archived(summary):
                continue
            membership = set(summary.source_message_ids)
            slots = [s for s in projection["narrative_slots"]
                     if membership.issubset({r["uid"] for r in s["covered_sources"]})]
            if len(slots) != 1:
                projection_refuse("projection_placement", "/narrative_slots",
                                  "An active narrative requires one declared complete slot.")
            insertions.setdefault(slots[0]["gap"], []).append(summary)
            narrative_sources.update(membership)
        out = []
        for j in range(len(request_messages) + 1):
            for summary in sorted(insertions.get(j, []), key=lambda s: (s.anchor_index, s.uid)):
                out.append(provider_envelope(render_summary_message(summary, summary.anchor_index)))
            if j == len(request_messages):
                break
            row = projection["rows"][j]
            message = deepcopy(request_messages[j])
            origin, policy = row["origin"], row["policy"]
            if origin["kind"] == "source":
                rec = self.store.get(origin["source"]["uid"])
                if not rec.active:
                    action = "narrative" if rec.uid in narrative_sources else "archive"
                    if not policy[action]:
                        projection_refuse("projection_ineligible", "/rows/" + str(j),
                                          "The current view does not permit this archived presentation.")
                    continue
                if rec.body_compaction is not None:
                    if not policy["body"]:
                        projection_refuse("projection_ineligible", "/rows/" + str(j),
                                          "A body presentation is unsafe for this current view.")
                    message["content"] = self.store.render_body(rec)
            out.append(message)
        validation = validate_tool_protocol(out)
        if not out or not validation.valid:
            projection_refuse("projection_topology", "", "Projected output is not a closed nonempty request.")
        return out

    # ------------------------------------------------------------------ #
    # notices                                                             #
    # ------------------------------------------------------------------ #

    def notice_for_level(self, level: float) -> str:
        active = self.store.working_set_tokens()
        return build_notice(level, active, self.budget.usable_tokens)

    def overflow_notice(self, retries_left: int) -> str:
        active = self.store.working_set_tokens()
        return urgent_overflow_notice(active, self.budget.usable_tokens, retries_left)
