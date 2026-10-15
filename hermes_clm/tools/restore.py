"""clm_restore (spec §13) — budget-aware, with required-space reporting."""

from __future__ import annotations

from typing import Any, Dict

from ..policies import PolicyViolation, validate_operation
from ..telemetry import observe_reference_restore


def _refused_response(engine: Any, exc: PolicyViolation) -> Dict[str, Any]:
    """Structured refusal envelope with required-space reporting (spec §13)."""
    response = {"ok": False, "error": exc.code, "message": exc.message,
                "message_ids": exc.message_ids}
    if exc.required_tokens is not None or exc.available_tokens is not None:
        response.update(required_tokens=exc.required_tokens, available_tokens=exc.available_tokens)
    if exc.code == "budget_exceeded":
        response["tokens_required"] = exc.required_tokens
        response["headroom_tokens"] = engine.budget.headroom(engine.store)
        response["hint"] = ("Free space first: compact or archive other "
                            "stale material, then retry.")
    return response


@observe_reference_restore
def tool_restore(engine: Any, args: Dict[str, Any]) -> Dict[str, Any]:
    if 'archive_refs' in args:
        from ..retrieval.restore import OMITTED
        if set(args) - {'archive_refs', 'reason', 'return_plan'}:
            # Pinned behavior: malformed reference invocations raise out of the
            # raw tool entry (the host dispatcher converts for the model).
            raise PolicyViolation('invalid_restore_arguments', 'Archive references cannot be combined with legacy selectors or host fields')
        try:
            return engine.apply_restore(archive_refs=args['archive_refs'], reason=args.get('reason', OMITTED),
                                        return_plan=args.get('return_plan'))
        except PolicyViolation as exc:
            return _refused_response(engine, exc)
    message_ids = [str(m) for m in (args.get("message_ids") or [])]
    summary_id = str(args.get("summary_id") or "")
    restore_sources = bool(args.get("restore_sources") or False)
    if not message_ids and not summary_id:
        return {"ok": False, "error": "empty_operation",
                "message": "supply message_ids or summary_id (or both)"}
    targets = list(message_ids)
    if summary_id:
        summary = engine.store.summary_by_prefix(summary_id)
        if summary is None:
            return {"ok": False, "error": "unknown_summary",
                    "message": f"no such summary: {summary_id}"}
        if restore_sources:
            targets.extend(summary.source_message_ids)
    if not targets:
        return {"ok": False, "error": "empty_operation",
                "message": "nothing to restore (summary has no sources and no "
                           "message_ids given)"}
    try:
        result = engine.apply_restore(targets, summary_id=summary_id,
                                      restore_sources=restore_sources, return_plan=args.get('return_plan'))
        if summary_id:
            result["summary_id"] = summary_id
            result["note"] = ("The summary is deactivated; original messages "
                              "are active again.")
        return result
    except PolicyViolation as exc:
        return _refused_response(engine, exc)
