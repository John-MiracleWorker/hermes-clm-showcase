"""Canonical message envelopes and exact-ID tool protocol validation.

The compatible-provider adapter is deliberately a denylist of known persistence
fields, not a provider-field allowlist. Unknown provider extensions survive.
No role adaptation is performed: the staged host/template owns that decision.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

# Mirrors staged Hermes agent/message_metadata.py at d9f0a399; independent of
# optional host imports so offline and in-host rendering have the same contract.
HOST_PERSISTENCE_FIELDS = frozenset({
    "timestamp", "display_kind", "display_metadata", "_row_id", "_submit_row_session_id",
    "_merged_turn_prefix", "message_uid", "_absorbed_message_uids", "_tool_call_uids",
    "_tool_call_uid", "_db_row_snapshot", "_canonical_row",
})
CLM_PERSISTENCE_FIELDS = frozenset({
    "uid", "protected", "active", "archived", "metadata", "summary_parent_ids",
    "source", "created_at", "token_count", "index", "toolobs_ref", "toolobs_chars",
    "permanent_reasons", "temporary_reasons", "body_compaction", "body_compaction_history",
})


def canonical_envelope(message: dict) -> dict:
    """Deep-copy every supplied field without coercing content or call data."""
    if not isinstance(message, dict):
        raise TypeError("message must be a dict")
    validate_call_container(message)
    return deepcopy(message)


def provider_envelope(message: dict) -> dict:
    """Render a detached wire view, removing only identified internal metadata."""
    return deepcopy({k: v for k, v in message.items()
                     if k not in HOST_PERSISTENCE_FIELDS | CLM_PERSISTENCE_FIELDS
                     and not k.startswith("_clm_")})


def envelope_digest(message: dict) -> str:
    # JSON preserves null/list/string distinctions and the exact argument string.
    encoded = json.dumps(message, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def envelope_uid(index: int, message: dict) -> str:
    # Full provider payload identity; persistence-only metadata can be absent on
    # host request copies and must not invalidate an archive/summary edit.
    return "msg_v2_" + hashlib.sha256(
        f"{index}|{envelope_digest(provider_envelope(message))}".encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ToolGroup:
    assistant_uid: str
    source_uids: tuple[str, ...]
    call_ids: tuple[str, ...]
    result_uids: tuple[str, ...]
    complete: bool
    source_indices: tuple[int, ...]


@dataclass(frozen=True)
class ProtocolViolation:
    code: str
    message_index: int
    call_id: str | None = None
    detail: str = ""


@dataclass(frozen=True)
class ProtocolValidation:
    valid: bool
    violations: tuple[ProtocolViolation, ...]
    groups: tuple[ToolGroup, ...] = ()


class ProtocolError(ValueError):
    def __init__(self, validation: ProtocolValidation):
        self.validation = validation
        self.code = validation.violations[0].code
        super().__init__("invalid tool protocol: " + ", ".join(v.code for v in validation.violations))


class RecoveryRequired(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__("canonical replay/recovery required: " + code)


def validate_call_container(message: dict, index: int = 0) -> None:
    """Optional calls have three compatible forms: absent, null, or a list."""
    value = message.get("tool_calls")
    if value is not None and not isinstance(value, list):
        raise ProtocolError(ProtocolValidation(
            False, (ProtocolViolation("invalid_tool_calls", index),)))


def _analyze(messages: list[dict], allow_pending: bool) -> ProtocolValidation:
    violations: list[ProtocolViolation] = []
    builders: list[dict[str, Any]] = []
    calls: dict[str, dict] = {}
    results: set[str] = set()
    pending: set[str] = set()

    def violation(code, i, cid=None, detail=""):
        violations.append(ProtocolViolation(code, i, cid, detail))

    for i, message in enumerate(messages):
        if not isinstance(message, dict):
            violation("invalid_message", i)
            continue
        role = message.get("role")
        if not isinstance(role, str) or not role:
            violation("missing_role", i)
        tool_calls = message.get("tool_calls")
        if tool_calls is not None and not isinstance(tool_calls, list):
            violation("invalid_tool_calls", i)
            continue
        # Only tool results can follow an unresolved assistant emission. Pending
        # execution is valid only at the end of stored history, never midstream.
        if pending and role != "tool":
            violation("interrupted_tool_group", i, detail="non-result before all calls completed")
        if tool_calls:
            if role != "assistant":
                violation("calls_on_non_assistant", i)
            builder = {"assistant": i, "ids": [], "results": [], "valid_ids": True}
            builders.append(builder)
            for call in tool_calls:
                cid = call.get("id") if isinstance(call, dict) else None
                if not isinstance(cid, str) or not cid:
                    violation("missing_call_id", i)
                    builder["valid_ids"] = False
                    continue
                builder["ids"].append(cid)
                if cid in calls:
                    violation("duplicate_call_id", i, cid)
                    builder["valid_ids"] = False
                    continue
                calls[cid] = builder
                pending.add(cid)
                function = call.get("function")
                if (call.get("type") != "function" or not isinstance(function, dict)
                        or not isinstance(function.get("name"), str) or not function.get("name")
                        or not isinstance(function.get("arguments"), str)):
                    violation("invalid_call_envelope", i, cid)
        if role == "tool":
            cid = message.get("tool_call_id")
            if not isinstance(cid, str) or not cid:
                violation("missing_result_id", i)
            elif cid not in calls:
                violation("orphan_result", i, cid)
            elif cid in results:
                violation("duplicate_result_id", i, cid)
            else:
                calls[cid]["results"].append(i)
                results.add(cid)
                pending.discard(cid)
        elif "tool_call_id" in message:
            violation("result_on_non_tool", i, message.get("tool_call_id"))
    if pending and not allow_pending:
        for cid in sorted(pending):
            violation("unresolved_calls", calls[cid]["assistant"], cid)

    def uid(i):
        return messages[i].get("_clm_uid") or envelope_uid(i, messages[i])

    groups = []
    for b in builders:
        indices = (b["assistant"], *sorted(b["results"]))
        groups.append(ToolGroup(uid(b["assistant"]), tuple(uid(i) for i in indices),
                                tuple(b["ids"]), tuple(uid(i) for i in sorted(b["results"])),
                                b["valid_ids"] and all(cid in results for cid in b["ids"]),
                                indices))
    return ProtocolValidation(not violations, tuple(violations), tuple(groups))


def tool_groups(messages: list[dict]) -> list[ToolGroup]:
    """Describe groups, including pending groups; validate separately before use."""
    return list(_analyze(messages, allow_pending=True).groups)


def validate_tool_protocol(messages: list[dict], *, allow_pending=False) -> ProtocolValidation:
    return _analyze(messages, allow_pending)


COORDINATE_ORIGIN = {"version": 1, "mode": "canonical_native", "producer": "source_projection_v1"}
PROJECTION_ADAPTATIONS = frozenset({"sidecar", "user_context", "reasoning", "strict_tools",
    "empty_payload", "expired_confirmation", "recovery", "moa_context", "bookkeeping"})


def projection_refuse(code, path="", message="Source identity cannot be admitted."):
    # Native hooks are host-only; standalone v4 never needs the optional host import.
    from agent.context_engine import SourceProjectionError
    raise SourceProjectionError({"code": code, "path": path, "message": message})


def validate_coordinate_origin(value):
    if (type(value) is not dict or set(value) != set(COORDINATE_ORIGIN)
            or type(value["version"]) is not int or value["version"] != 1
            or type(value["mode"]) is not str or value["mode"] != "canonical_native"
            or type(value["producer"]) is not str or value["producer"] != "source_projection_v1"):
        raise ValueError("invalid coordinate_origin")


def validate_json(value, path="", ancestors=None):
    """Exact recursive JSON, including UTF-8 and cycle checks, before hashing."""
    ancestors = set() if ancestors is None else ancestors
    t = type(value)
    if value is None or t in (bool, int):
        return
    if t is str:
        try:
            value.encode("utf-8")
        except UnicodeError:
            projection_refuse("projection_schema", path, "Invalid UTF-8 string.")
        return
    if t is float and math.isfinite(value):
        return
    if t not in (dict, list) or id(value) in ancestors:
        projection_refuse("projection_schema", path, "Expected finite acyclic JSON.")
    ancestors.add(id(value))
    try:
        for key, item in (value.items() if t is dict else enumerate(value)):
            if t is dict:
                if type(key) is not str:
                    projection_refuse("projection_schema", path, "JSON keys must be strings.")
                validate_json(key, path, ancestors)
            part = str(key).replace("~", "~0").replace("/", "~1")
            validate_json(item, path + "/" + part, ancestors)
    finally:
        ancestors.remove(id(value))


def validate_native_protocol(messages):
    """Canonical pending calls may await host recovery; wire calls must close."""
    result = validate_tool_protocol(messages, allow_pending=True)
    violations = tuple(v for v in result.violations if v.code != "interrupted_tool_group")
    return ProtocolValidation(not violations, violations, result.groups)


class _ProjectionValidator:
    def __init__(self, projection, canonical, wire, incoming, retained):
        self.p, self.canonical, self.wire = projection, canonical, wire
        self.incoming, self.retained = incoming, retained

    @staticmethod
    def reject(path, message, code="projection_schema"):
        projection_refuse(code, path, message)

    def obj(self, value, fields, path):
        if type(value) is not dict or set(value) != set(fields):
            self.reject(path, "Missing or unknown object fields.")

    def string(self, value, path, *, choices=None, digest=False):
        if type(value) is not str or not value:
            self.reject(path, "Expected a nonempty string.")
        validate_json(value, path)
        if choices is not None and value not in choices:
            self.reject(path, "Unsupported value.")
        if digest and (len(value) != 64 or any(c not in "0123456789abcdef" for c in value)):
            self.reject(path, "Expected a lowercase SHA-256 digest.")

    def index(self, value, bound, path, *, nullable=False):
        if nullable and value is None:
            return
        if type(value) is not int or not 0 <= value < bound:
            self.reject(path, "Index is outside its exact coordinate range.")

    def array(self, value, path, *, nonempty=False):
        if type(value) is not list or (nonempty and not value):
            self.reject(path, "Expected an exact list.")

    def ref(self, value, path):
        self.obj(value, ("index", "uid"), path)
        self.index(value["index"], len(self.canonical), path + "/index")
        self.string(value["uid"], path + "/uid")

    def refs(self, value, path):
        self.array(value, path, nonempty=True)
        for i, ref in enumerate(value):
            self.ref(ref, path + "/" + str(i))
        indices = [r["index"] for r in value]
        if indices != sorted(set(indices)):
            self.reject(path, "References must be unique and ascending.")

    def schema(self):
        p = self.p
        self.obj(p, ("version", "mode", "conversation_digest", "request_digest", "incoming_index",
                     "sources", "rows", "omissions", "narrative_slots"), "")
        if type(p["version"]) is not int or p["version"] != 1:
            self.reject("/version", "Unsupported projection version.")
        self.string(p["mode"], "/mode", choices={"native"})
        for key in ("conversation_digest", "request_digest"):
            self.string(p[key], "/" + key, digest=True)
        self.index(p["incoming_index"], len(self.canonical), "/incoming_index", nullable=True)
        self.array(p["sources"], "/sources")
        for i, source in enumerate(p["sources"]):
            path = "/sources/" + str(i)
            self.obj(source, ("index", "uid", "exact_digest"), path)
            self.index(source["index"], len(self.canonical), path + "/index")
            self.string(source["uid"], path + "/uid")
            self.string(source["exact_digest"], path + "/exact_digest", digest=True)
        self.array(p["rows"], "/rows", nonempty=True)
        for i, row in enumerate(p["rows"]):
            path = "/rows/" + str(i)
            self.obj(row, ("request_index", "wire_digest", "origin", "policy"), path)
            self.index(row["request_index"], len(self.wire), path + "/request_index")
            self.string(row["wire_digest"], path + "/wire_digest", digest=True)
            origin = row["origin"]
            if type(origin) is not dict:
                self.reject(path + "/origin", "Expected an origin object.")
            kind = origin.get("kind")
            self.string(kind, path + "/origin/kind", choices={"source", "ephemeral", "generated"})
            op = path + "/origin"
            if kind == "source":
                self.obj(origin, ("kind", "source", "adaptations"), op)
                self.ref(origin["source"], op + "/source")
                adaptations = origin["adaptations"]
                self.array(adaptations, op + "/adaptations")
                for j, a in enumerate(adaptations):
                    self.string(a, op + "/adaptations/" + str(j), choices=PROJECTION_ADAPTATIONS)
                if len(adaptations) != len(set(adaptations)):
                    self.reject(op + "/adaptations", "Duplicate adaptation.")
            elif kind == "ephemeral":
                self.obj(origin, ("kind", "purpose", "occurrence"), op)
                self.string(origin["purpose"], op + "/purpose", choices={"system_frame", "prefill", "suffix", "notice"})
                if type(origin["occurrence"]) is not int or origin["occurrence"] < 0:
                    self.reject(op + "/occurrence", "Expected a nonnegative occurrence.")
            else:
                self.obj(origin, ("kind", "purpose", "parents", "association", "effect"), op)
                self.string(origin["purpose"], op + "/purpose", choices={"recovery"})
                self.refs(origin["parents"], op + "/parents")
                a = origin["association"]
                self.obj(a, ("assistant", "call_id", "function_name"), op + "/association")
                self.ref(a["assistant"], op + "/association/assistant")
                for key in ("call_id", "function_name"):
                    self.string(a[key], op + "/association/" + key)
                self.string(origin["effect"], op + "/effect", choices={"unknown", "none"})
            self.obj(row["policy"], ("essential", "archive", "body", "narrative"), path + "/policy")
            for key, value in row["policy"].items():
                if type(value) is not bool:
                    self.reject(path + "/policy/" + key, "Expected an exact boolean.")
        self.array(p["omissions"], "/omissions")
        for i, omission in enumerate(p["omissions"]):
            path = "/omissions/" + str(i)
            self.obj(omission, ("source", "reason", "group_sources"), path)
            self.ref(omission["source"], path + "/source")
            self.string(omission["reason"], path + "/reason", choices={"interrupted_read_only", "dangling_read_only"})
            self.refs(omission["group_sources"], path + "/group_sources")
        self.array(p["narrative_slots"], "/narrative_slots")
        for i, slot in enumerate(p["narrative_slots"]):
            path = "/narrative_slots/" + str(i)
            self.obj(slot, ("gap", "left", "right", "covered_sources"), path)
            self.index(slot["gap"], len(self.wire) + 1, path + "/gap")
            for key in ("left", "right"):
                self.index(slot[key], len(self.wire), path + "/" + key, nullable=True)
            self.refs(slot["covered_sources"], path + "/covered_sources")

    def binding(self):
        p = self.p
        if len(p["sources"]) != len(self.canonical) or len(p["rows"]) != len(self.wire):
            self.reject("", "Source or request ledger is incomplete.", "projection_coverage")
        for key, messages in (("conversation_digest", self.canonical), ("request_digest", self.wire)):
            if p[key] != envelope_digest(messages):
                self.reject("/" + key, "Complete message digest does not match.", "projection_binding")
        for i, source in enumerate(p["sources"]):
            if (source["index"] != i or source["uid"] != envelope_uid(i, self.canonical[i])
                    or source["exact_digest"] != envelope_digest(self.canonical[i])):
                self.reject("/sources/" + str(i), "Canonical occurrence binding does not match.", "projection_binding")
        for i, row in enumerate(p["rows"]):
            if row["request_index"] != i or row["wire_digest"] != envelope_digest(self.wire[i]):
                self.reject("/rows/" + str(i), "Request occurrence binding does not match.", "projection_binding")
        anchor = p["incoming_index"]
        if ((anchor is None and self.incoming is not None)
                or (anchor is not None and envelope_digest(self.incoming) != envelope_digest(self.canonical[anchor]))):
            self.reject("/incoming_index", "Incoming occurrence does not match.", "projection_binding")
        def bound(ref, path):
            if ref["uid"] != p["sources"][ref["index"]]["uid"]:
                self.reject(path, "Reference UID does not match its occurrence.", "projection_binding")
        for i, row in enumerate(p["rows"]):
            o = row["origin"]
            if o["kind"] == "source":
                bound(o["source"], "/rows/" + str(i) + "/origin/source")
            elif o["kind"] == "generated":
                for ref in [*o["parents"], o["association"]["assistant"]]:
                    bound(ref, "/rows/" + str(i) + "/origin")
        for key in ("omissions", "narrative_slots"):
            for i, item in enumerate(p[key]):
                refs = item["group_sources"] + [item["source"]] if key == "omissions" else item["covered_sources"]
                for ref in refs:
                    bound(ref, "/" + key + "/" + str(i))
        validate_source_conflicts(self.canonical, self.retained)

    def semantics(self):
        p = self.p
        represented, omitted, ephemeral = {}, {}, set()
        for j, row in enumerate(p["rows"]):
            origin = row["origin"]
            if origin["kind"] == "source":
                index = origin["source"]["index"]
                if index in represented:
                    self.reject("/rows/" + str(j), "Duplicate canonical occurrence.", "projection_coverage")
                represented[index] = row
            elif origin["kind"] == "ephemeral":
                key = (origin["purpose"], origin["occurrence"])
                if key in ephemeral:
                    self.reject("/rows/" + str(j), "Duplicate ephemeral occurrence.", "projection_coverage")
                ephemeral.add(key)
        for j, omission in enumerate(p["omissions"]):
            index = omission["source"]["index"]
            if index in represented or index in omitted:
                self.reject("/omissions/" + str(j), "Duplicate source disposition.", "projection_coverage")
            omitted[index] = omission
        if set(represented) | set(omitted) != set(range(len(self.canonical))):
            self.reject("", "Every canonical occurrence requires one disposition.", "projection_coverage")
        native = validate_native_protocol(self.canonical)
        wire = validate_tool_protocol(self.wire)
        if not native.valid or not wire.valid:
            self.reject("", "Tool associations are not closed on the request.", "projection_topology")
        by_source = {i: g for g in native.groups for i in g.source_indices}
        from agent.tool_result_classification import tool_may_have_side_effect
        from agent.replay_cleanup import is_interrupted_tool_result, _INTERRUPTED_NOTICES, _DANGLING_NOTICES
        for i, omission in omitted.items():
            g = by_source.get(i)
            indices = [r["index"] for r in omission["group_sources"]]
            if (g is None or indices != list(g.source_indices) or not set(indices).issubset(omitted)
                    or any(omitted[k]["group_sources"] != omission["group_sources"]
                           or omitted[k]["reason"] != omission["reason"] for k in indices)):
                self.reject("/omissions", "Omission must cover its entire exact tool group.", "projection_topology")
            calls = self.canonical[g.source_indices[0]].get("tool_calls") or []
            interrupted = any(is_interrupted_tool_result(self.canonical[k].get("content")) for k in g.source_indices[1:])
            if (any(tool_may_have_side_effect(c["function"]["name"]) for c in calls)
                    or (omission["reason"] == "interrupted_read_only" and not interrupted)
                    or (omission["reason"] == "dangling_read_only" and (g.complete or len(g.source_indices) != 1))):
                self.reject("/omissions", "Unsupported read-only omission.", "projection_topology")
        for j, row in enumerate(p["rows"]):
            origin, policy = row["origin"], row["policy"]
            path = "/rows/" + str(j)
            if policy["essential"] and any(policy[k] for k in ("archive", "body", "narrative")):
                self.reject(path + "/policy", "Essential rows cannot be edited.", "projection_ineligible")
            if origin["kind"] != "source":
                if policy != {"essential": True, "archive": False, "body": False, "narrative": False}:
                    self.reject(path + "/policy", "Generated and ephemeral rows are essential.", "projection_ineligible")
                if origin["kind"] == "generated":
                    a = origin["association"]
                    ai = a["assistant"]["index"]
                    parents = [r["index"] for r in origin["parents"]]
                    calls = self.canonical[ai].get("tool_calls") or []
                    matches = [c for c in calls if c["id"] == a["call_id"] and c["function"]["name"] == a["function_name"]]
                    notices = _DANGLING_NOTICES
                    effect = "unknown" if tool_may_have_side_effect(a["function_name"]) else "none"
                    m = self.wire[j]
                    if (ai not in parents or len(matches) != 1 or m.get("role") != "tool"
                            or m.get("tool_call_id") != a["call_id"] or m.get("name") != a["function_name"]
                            or origin["effect"] != effect or m.get("effect_disposition") != effect
                            or m.get("content") != notices[0 if effect == "unknown" else 1]):
                        self.reject(path + "/origin", "Generated recovery association does not match.", "projection_topology")
                continue
            i = origin["source"]["index"]
            adaptations = origin["adaptations"]
            g = by_source.get(i)
            if ((g is not None and not g.complete) or set(adaptations) & {"expired_confirmation", "recovery", "sidecar", "user_context", "moa_context"}):
                if not policy["essential"]:
                    self.reject(path + "/policy", "Safety and pending views must be essential.", "projection_ineligible")
            if policy["body"] and (self.canonical[i].get("role") != "tool" or g is None or not g.complete
                    or type(self.wire[j].get("content")) is not str or not self.wire[j]["content"]
                    or set(adaptations) - {"bookkeeping"}
                    or self.wire[j].get("tool_call_id") != self.canonical[i].get("tool_call_id")):
                self.reject(path + "/policy/body", "This view is ineligible for body edits.", "projection_ineligible")
        for g in native.groups:
            members = [represented[i] for i in g.source_indices if i in represented]
            for edit in ("archive", "narrative"):
                if members and len({r["policy"][edit] for r in members}) != 1:
                    self.reject("/rows", "Whole-group eligibility must be uniform.", "projection_ineligible")
            if any(r["policy"]["body"] for r in members):
                if len(members) != len(g.source_indices) or any(r["policy"]["essential"] for r in members):
                    self.reject("/rows", "Body edits require a complete nonessential group.", "projection_ineligible")
        used_gaps, covered = set(), set()
        for j, slot in enumerate(p["narrative_slots"]):
            gap = slot["gap"]
            indices = [r["index"] for r in slot["covered_sources"]]
            if (gap in used_gaps or slot["left"] != (gap - 1 if gap else None)
                    or slot["right"] != (gap if gap < len(self.wire) else None)
                    or covered.intersection(indices)):
                self.reject("/narrative_slots/" + str(j), "Invalid or duplicate narrative gap.", "projection_placement")
            used_gaps.add(gap)
            covered.update(indices)
            for i in indices:
                g = by_source.get(i)
                if (i not in represented or not represented[i]["policy"]["narrative"]
                        or (g is not None and (not g.complete or not set(g.source_indices).issubset(indices)))):
                    self.reject("/narrative_slots/" + str(j), "Slot requires eligible complete groups.", "projection_placement")
            if any(min(g.source_indices) < gap <= max(g.source_indices) for g in wire.groups):
                self.reject("/narrative_slots/" + str(j), "Narrative gap splits a tool envelope.", "projection_placement")
        return represented


def validate_source_conflicts(canonical, retained):
    for i, message in enumerate(canonical):
        uid = envelope_uid(i, message)
        record = retained.get(uid)
        if record is not None and envelope_digest(record.envelope) != envelope_digest(message):
            projection_refuse("source_envelope_conflict", "/sources/" + str(i),
                              "A retained occurrence has a different exact envelope.")


def validate_source_projection(projection, canonical, wire, incoming, retained):
    if projection is None:
        projection_refuse("projection_missing", "", "Native selection requires a source projection.")
    for name, rows in (("conversation_messages", canonical), ("request_messages", wire)):
        if type(rows) is not list or any(type(m) is not dict for m in rows) or (name == "request_messages" and not rows):
            projection_refuse("projection_schema", "", "Expected exact message lists.")
        validate_json(rows)
    if incoming is not None:
        validate_json(incoming)
    validator = _ProjectionValidator(projection, canonical, wire, incoming, retained)
    validator.schema()
    validator.binding()
    return validator.semantics()
