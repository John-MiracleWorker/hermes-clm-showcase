"""clm_archive (spec §12)."""

from __future__ import annotations

from typing import Any, Dict


def tool_archive(engine: Any, args: Dict[str, Any]) -> Dict[str, Any]:
    message_ids = [str(m) for m in (args.get("message_ids") or [])]
    reason = str(args.get("reason") or "")
    if not message_ids:
        return {"ok": False, "error": "empty_operation",
                "message": "message_ids is required and must be non-empty"}
    metadata = {key: args[key] for key in ('map_path', 'map_description', 'keywords') if key in args}
    result = engine.apply_archive(message_ids, reason=reason, **metadata)
    result["message_ids"] = result["archived_ids"]
    result["restore_hint"] = ("Use clm_restore with these message_ids to bring "
                              "them back. Archived content is never destroyed.")
    return result
