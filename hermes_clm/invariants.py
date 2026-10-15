"""Pure, detached checks and plans for canonical residency and known lineage."""
from copy import deepcopy
from dataclasses import dataclass

from .models import ContextMessage, ContextSummary, validate_body_compaction
from .protocol import (ProtocolError, RecoveryRequired, envelope_uid,
                       validate_tool_protocol, validate_native_protocol)


class InvariantError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class InvariantSnapshot:
    messages: dict
    order: tuple
    by_index: dict
    summaries: dict
    recovery_required: str | None
    revision: int
    summary_seq: int
    toolobs_seq: int
    canonical_native: bool = False


@dataclass(frozen=True)
class ReconciliationPlan:
    residency: dict
    retire_bodies: tuple
    retire_summaries: tuple


def capture(store):
    return InvariantSnapshot(
        deepcopy({u: r.to_dict() for u, r in store.messages.items()}),
        tuple(store.order), deepcopy(store._by_index),
        deepcopy({u: s.to_dict() for u, s in store.summaries.items()}),
        store.recovery_required, store.revision, store._summary_seq,
        store._toolobs_seq, store.coordinate_origin is not None)


def protocol(snapshot, *, allow_pending=True):
    messages = [dict(snapshot.messages[u]["envelope"], _clm_uid=u) for u in snapshot.order]
    if snapshot.canonical_native and allow_pending:
        return validate_native_protocol(messages)
    return validate_tool_protocol(messages, allow_pending=allow_pending)


def source_groups(snapshot, sources):
    """A historic join is checked against its own retained original branch."""
    records = snapshot.messages
    if (not isinstance(sources, list) or not sources
            or any(not isinstance(u, str) or u not in records for u in sources)
            or len(sources) != len(set(sources))):
        raise InvariantError("invalid_derivation_sources")
    indices = [records[u]["index"] for u in sources]
    if indices != sorted(set(indices)):
        raise InvariantError("unordered_or_mixed_branch_sources")
    envelopes = [records[u]["envelope"] for u in sources]
    if any(not isinstance(e, dict) for e in envelopes):
        raise InvariantError("original_envelopes_unavailable")
    result = validate_tool_protocol([dict(e, _clm_uid=u) for e, u in zip(envelopes, sources)])
    if not result.valid:
        raise InvariantError("incomplete_source_tool_groups")
    return list(result.groups)


def validate_body(snapshot, uid, overlay, *, current):
    validate_body_compaction(overlay)
    if uid not in snapshot.messages or snapshot.messages[uid]["role"] != "tool" or "source_uids" in overlay:
        raise InvariantError("invalid_body_compaction_contract")
    membership = overlay["restore_group_uids"]
    derivation = overlay["derivation_source_uids"]
    source_groups(snapshot, derivation)
    groups = source_groups(snapshot, membership)
    if (len(groups) != 1 or uid not in groups[0].result_uids
            or membership != list(groups[0].source_uids) or not set(membership).issubset(derivation)):
        raise InvariantError("incomplete_body_restore_group")
    if current:
        group = next((g for g in protocol(snapshot).groups if uid in g.result_uids), None)
        if (not set(derivation).issubset(snapshot.order) or group is None
                or not group.complete or membership != list(group.source_uids)):
            raise InvariantError("invalid_current_body_sources")


def validate(snapshot, *, allow_pending, diagnostic=False):
    if snapshot.recovery_required:
        raise RecoveryRequired(snapshot.recovery_required)
    for value in (snapshot.revision, snapshot.summary_seq, snapshot.toolobs_seq):
        if type(value) is not int or value < 0:
            raise InvariantError("invalid_allocation_checkpoint")
    # Summary allocation owns these IDs, including inactive and historic ones.
    # Tool observation refs are supplied externally, not allocated by the store.
    for uid in snapshot.summaries:
        suffix = uid.removeprefix("clm_summary_")
        if (uid.startswith("clm_summary_") and suffix.isascii()
                and suffix.isdecimal() and int(suffix) > snapshot.summary_seq):
            raise InvariantError("stale_summary_allocation_checkpoint")
    records, order = snapshot.messages, snapshot.order
    if len(order) != len(set(order)) or any(u not in records for u in order):
        raise InvariantError("invalid_canonical_order")
    if ([records[u]["index"] for u in order] != list(range(len(order)))
            or snapshot.by_index != {i: u for i, u in enumerate(order)}):
        raise InvariantError("invalid_canonical_order")
    for uid, record in records.items():
        ContextMessage.from_dict(record)
        e = record["envelope"]
        if (not isinstance(e, dict) or not isinstance(e.get("role"), str) or not e["role"]
                or record["uid"] != uid or envelope_uid(record["index"], e) != uid
                or record["role"] != e["role"] or record["active"] == record["archived"]):
            raise InvariantError("invalid_canonical_identity")
        content = e.get("content")
        if isinstance(content, list):
            content = "\n".join(str(p.get("text", "")) if isinstance(p, dict) else p
                                for p in content if isinstance(p, str)
                                or isinstance(p, dict) and p.get("type") == "text")
        elif not isinstance(content, str):
            content = "" if content is None else str(content)
        if record["content"] != content:
            raise InvariantError("invalid_indexing_content")
        if uid not in order and (record["active"] or not record["archived"]):
            raise InvariantError("active_ghost_outside_canonical_order")
    first_user = next((u for u in order if records[u]["role"] == "user"), None)
    for uid in order:
        r = records[uid]
        if (uid == first_user or r["role"] in {"system", "developer"} or "pinned" in r["permanent_reasons"]):
            if not r["protected"] or not r["active"]:
                raise InvariantError("missing_permanent_protection")
    result = protocol(snapshot, allow_pending=allow_pending)
    if not result.valid and not diagnostic:
        raise ProtocolError(result)
    for group in result.groups:
        flags = {(records[u]["active"], records[u]["archived"]) for u in group.source_uids}
        if len(flags) != 1 or (not group.complete and flags != {(True, False)}):
            raise InvariantError("inconsistent_tool_group_state")
    for uid, r in records.items():
        if r["body_compaction"] is not None:
            validate_body(snapshot, uid, r["body_compaction"], current=True)
        for overlay in r["body_compaction_history"]:
            validate_body(snapshot, uid, overlay, current=False)
    for uid, summary in snapshot.summaries.items():
        ContextSummary.from_dict(summary)
        if summary["uid"] != uid or summary.get("version") != 1 or "derivation_source_uids" not in summary:
            raise InvariantError("missing_narrative_lineage")
        source_groups(snapshot, summary["source_message_ids"])
        source_groups(snapshot, summary["derivation_source_uids"])
        if summary["anchor_index"] != records[summary["source_message_ids"][0]]["index"]:
            raise InvariantError("invalid_narrative_anchor")
        if summary["revision"] < 1:
            raise InvariantError("invalid_narrative_revision")
        if summary["active"] and not set(summary["derivation_source_uids"]).issubset(order):
            raise InvariantError("active_summary_from_historic_branch")


def reconcile_canonical(previous, candidate):
    """Only proven complete identical groups carry an earlier residency choice."""
    prior = protocol(previous)
    # An earlier diagnostic pending call can report an interruption at the
    # start of a separately valid sibling. Check that sibling's own complete
    # branch before preserving it; the preceding interruption is not its input.
    invalid_indices = {v.message_index for v in prior.violations
                       if v.code != "interrupted_tool_group"}
    proven = {}
    for g in prior.groups:
        flags = {(previous.messages[u]["active"], previous.messages[u]["archived"]) for u in g.source_uids}
        if g.complete and len(flags) == 1 and not invalid_indices.intersection(g.source_indices):
            try:
                source_groups(previous, list(g.source_uids))
            except ValueError:
                continue
            proven[g.source_uids] = next(iter(flags))
    residency = {u: ((previous.messages[u]["active"], previous.messages[u]["archived"])
                     if u in previous.order else (True, False)) for u in candidate.order}
    groups = protocol(candidate).groups
    for g in groups:
        flags = proven.get(g.source_uids, (True, False)) if g.complete else (True, False)
        residency.update({u: flags for u in g.source_uids})
    by_result = {u: g for g in groups for u in g.result_uids}
    retired = []
    for uid, rec in candidate.messages.items():
        overlay = rec["body_compaction"]
        if overlay is not None:
            g = by_result.get(uid)
            if (g is None or not g.complete or list(g.source_uids) != overlay["restore_group_uids"]
                    or not set(overlay["derivation_source_uids"]).issubset(candidate.order)):
                retired.append(uid)
    summaries = tuple(u for u, s in candidate.summaries.items()
                      if not set(s["derivation_source_uids"]).issubset(candidate.order))
    return ReconciliationPlan(residency, tuple(retired), summaries)


def derive_inputs(snapshot, membership):
    source_groups(snapshot, membership)
    known = set(membership)
    for uid in membership:
        overlay = snapshot.messages[uid]["body_compaction"]
        if overlay is not None:
            known.update(overlay["derivation_source_uids"])
    for summary in snapshot.summaries.values():
        members = summary["source_message_ids"]
        if (summary["active"] and set(members).intersection(membership)
                and set(summary["derivation_source_uids"]).issubset(snapshot.order)
                and all(u in snapshot.order and snapshot.messages[u]["archived"] for u in members)):
            known.update(summary["derivation_source_uids"])
    result = sorted(known, key=lambda u: snapshot.messages[u]["index"])
    source_groups(snapshot, result)
    return result
