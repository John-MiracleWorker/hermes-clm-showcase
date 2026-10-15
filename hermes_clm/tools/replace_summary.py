"""clm_replace_summary (spec §14)."""

from __future__ import annotations

from typing import Any, Dict


def tool_replace_summary(engine: Any, args: Dict[str, Any]) -> Dict[str, Any]:
    summary_id = str(args.get("summary_id") or "")
    replacement = str(args.get("replacement") or "")
    if not summary_id:
        return {"ok": False, "error": "empty_operation",
                "message": "summary_id is required"}
    return engine.apply_replace_summary(summary_id, replacement)
