"""clm_compact (spec §11)."""

from __future__ import annotations

from typing import Any, Dict


def tool_compact(engine: Any, args: Dict[str, Any]) -> Dict[str, Any]:
    message_ids = [str(m) for m in (args.get("message_ids") or [])]
    summary = str(args.get("summary") or "")
    reason = str(args.get("reason") or "")
    if not message_ids:
        return {"ok": False, "error": "empty_operation",
                "message": "message_ids is required and must be non-empty"}
    metadata = {key: args[key] for key in ('map_path', 'map_description', 'keywords') if key in args}
    result = engine.apply_compact(message_ids, summary, reason=reason, form=args.get("form", "summary"),
                                  **metadata)
    return result
