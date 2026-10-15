"""``/clm`` chat command + ``hermes clm`` inspection (spec §48/§49).

Read-only except ``restore`` (a legitimate CLM operation, still
reversible). Output is plain text for chat and CLI alike.
"""

import json
from typing import Any, Dict


def _fmt_tokens(n: Any) -> str:
    try:
        return f"{int(n):,}"
    except (TypeError, ValueError):
        return str(n)


def _status_text(engine) -> str:
    from hermes_clm.tools import dispatch  # native path, same boundary as the model
    s = json.loads(dispatch(engine, "clm_status", {}))
    if not s.get("ok"):
        return f"CLM status error: {s.get('message', s)}"
    lines = [
        f"Conversation: {s.get('conversation_id', '?')}",
        f"Context engine: CLM (rev {s.get('context_revision', 0)})",
        "",
        f"Working budget:    {_fmt_tokens(s.get('budget_tokens'))}",
        f"Active tokens:     {_fmt_tokens(s.get('active_tokens'))}",
        f"Utilization:       {100 * float(s.get('utilization', 0)):.1f}%",
        f"Headroom:          {_fmt_tokens(s.get('headroom_tokens'))}",
        f"Active messages:   {s.get('active_messages', 0)}",
        f"Archived messages: {s.get('archived_messages', 0)}",
        f"Synthetic summaries: {s.get('summaries', 0)}",
        f"Provider window:   {_fmt_tokens(s.get('provider_context_length'))} "
        f"(hard limit {_fmt_tokens(s.get('hard_limit_tokens'))})",
    ]
    m = s.get("metrics", {})
    if m.get("last_compaction_at"):
        lines.append(f"Last compaction:   {m['last_compaction_at']}")
    return "\n".join(lines)


def _active_text(engine) -> str:
    recs = engine.list_active()
    if not recs:
        return "Active set is empty."
    lines = [f"{r['uid']}  {r['role']:<9} prot={int(r['protected'])} "
             f"tok={_fmt_tokens(r['tokens'])} ({r['chars']} chars)"
             for r in recs]
    return f"{len(recs)} active records:\n" + "\n".join(lines)


def _archive_text(engine) -> str:
    recs = engine.list_archive()
    if not recs:
        return "Archive is empty (nothing compacted/archived yet)."
    lines = [f"{r['uid']}  {r['role']:<9} tok={_fmt_tokens(r['tokens'])} "
             f"({r['chars']} chars)" for r in recs]
    return f"{len(recs)} archived records:\n" + "\n".join(lines)


def _summary_text(engine, summary_id: str) -> str:
    s = engine.show_summary(summary_id)
    if not s.get("ok"):
        known = ", ".join(sorted(engine.store.summaries)[:10]) or "(none)"
        return f"no such summary: {summary_id} (known: {known})"
    return (f"{s['uid']}  rev {s['revision']}  by {s['created_by']}\n"
            f"sources: {', '.join(s['source_message_ids'])}\n"
            f"---\n{s['content']}")


def _events_text(engine, limit: int = 20) -> str:
    events = engine.recent_events(limit)
    if not events:
        return "No event log yet."
    pretty = [f"{e.get('timestamp', '?')[:19]}  {e.get('event', '?'):<10} "
              f"rev {e.get('context_revision_before', '?')}"
              f"→{e.get('context_revision_after', '?')} "
              f"by {e.get('initiator', '?')}" for e in events]
    return f"Last {len(events)} events:\n" + "\n".join(pretty)


def _restore_text(engine, summary_id: str) -> str:
    from hermes_clm.tools import dispatch
    res = json.loads(dispatch(engine, "clm_restore",
                              {"summary_id": summary_id,
                               "restore_sources": True}))
    if not res.get("ok"):
        return f"restore failed: {res.get('error')} — {res.get('message', '')}"
    return (f"restored {res.get('tokens_restored', 0)} tokens "
            f"(rev {res.get('context_revision')})")


def _resolve_engine(engine):
    if engine is not None:
        return engine
    # The plugin module is registered under a synthetic name that may contain
    # a hyphen — importlib (not the import statement) handles that.
    try:
        import importlib
        _mod = importlib.import_module("_hermes_user_context_engine.hermes-clm")
        active = _mod.get_active_engine()
        if active is not None:
            return active
    except Exception:
        pass
    raise RuntimeError(
        "CLM engine not loaded — set context.engine: hermes-clm in config.yaml")


def handle_clm_command(raw_args: str, engine) -> str:
    """Entry point for the /clm chat command."""
    engine = _resolve_engine(engine)
    parts = (raw_args or "").split()
    cmd = parts[0].lower() if parts else "status"
    try:
        if cmd == "status":
            return _status_text(engine)
        if cmd in ("active", "list-active"):
            return _active_text(engine)
        if cmd in ("archive", "list-archive"):
            return _archive_text(engine)
        if cmd in ("summary", "show-summary"):
            if len(parts) < 2:
                return "usage: /clm summary <id>"
            return _summary_text(engine, parts[1])
        if cmd == "events":
            return _events_text(engine)
        if cmd == "restore":
            if len(parts) < 2:
                return "usage: /clm restore <summary-id>"
            return _restore_text(engine, parts[1])
        if cmd == "reset-working-set":
            engine.reset_working_set()
            return "working set rebuilt from canonical history " \
                   f"(rev {engine.store.revision})"
    except Exception as exc:  # never crash the chat on an inspection command
        return f"CLM command error: {exc}"
    return "usage: /clm [status|active|archive|summary <id>|restore <id>|" \
           "events|reset-working-set]"
