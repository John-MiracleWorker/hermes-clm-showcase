"""clm_status (spec §10)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict


def tool_status(engine: Any, args: Dict[str, Any]) -> Dict[str, Any]:
    store = engine.store
    working = store.working_set_tokens()
    s = engine.budget.status(store, working)
    last = store.metrics.get("last_compaction_at")
    if last:
        s["last_compaction_at"] = last
    s["tokenizer_backend"] = engine.counter.backend_name
    # Addressable objects the model can operate on.
    s["active_uids"] = [r.uid for r in store.active_messages()]
    s["archived"] = [
        {"uid": r.uid, "role": r.role, "tokens": r.token_count,
         "chars": len(r.content), "protected": r.protected}
        for r in store.archived_messages()
    ]
    s["summary_details"] = [
        {"uid": srec.uid, "revision": srec.revision,
         "source_message_ids": srec.source_message_ids,
         "derivation_source_uids": srec.derivation_source_uids,
         "dependencies_current": set(srec.derivation_source_uids or []).issubset(store.order),
         "membership_current": set(srec.source_message_ids).issubset(store.order),
         "rendering_eligible": srec.active and store.is_summary_fully_archived(srec),
         "tokens": srec.token_count, "active": srec.active,
         "content_head": srec.content[:400]}
        for srec in sorted(store.summaries.values(), key=lambda x: x.anchor_index)
    ]
    # spec §10: "summaries" in the budget block stays the COUNT.
    s["protected_uids"] = [r.uid for r in store.messages.values() if r.protected]
    # Which conversation this state belongs to (useful for /clm status).
    s["conversation_id"] = getattr(engine, "_conversation_id", None)
    return deepcopy({"ok": True, **s})
