"""Core data model for the CLM context engine (spec §6).

All records are plain dataclasses with ``to_dict``/``from_dict`` round-trips so the
persistence layer can serialize them with the stdlib json module only.
"""

from __future__ import annotations

from copy import deepcopy
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .protocol import validate_call_container


def _object(value: Any, label: str) -> None:
    if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
        raise ValueError(f"{label} must be a JSON object")


def _strings(d: dict, fields: tuple, *, nullable=()) -> None:
    for key in fields:
        if key in d and not (key in nullable and d[key] is None) and not isinstance(d[key], str):
            raise ValueError(f"{key} must be a string")


def _integers(d: dict, fields: tuple, *, nullable=()) -> None:
    for key in fields:
        if key in d and not (key in nullable and d[key] is None):
            if type(d[key]) is not int or d[key] < 0:
                raise ValueError(f"{key} must be a nonnegative integer")


def _booleans(d: dict, fields: tuple) -> None:
    for key in fields:
        if key in d and type(d[key]) is not bool:
            raise ValueError(f"{key} must be a boolean")


def _string_list(value: Any, label: str) -> None:
    if not isinstance(value, list) or any(not isinstance(v, str) or not v for v in value):
        raise ValueError(f"{label} must be a list of nonempty strings")


def validate_metrics(value: Any) -> None:
    _object(value, "metrics")
    counters = {"total_events", "context_compactions", "tokens_removed",
                "summary_tokens_created", "context_archives", "context_restores",
                "tokens_restored", "context_rejections", "overflow_events",
                "emergency_host_compactions", "total_model_calls", "total_prompt_tokens",
                "total_completion_tokens", "last_compaction_tokens"}
    _integers(value, tuple(k for k in value if k in counters or k.startswith("thrash::")))
    _strings(value, ("last_compaction_at",))


def validate_body_compaction(value: Any) -> None:
    _object(value, "body compaction")
    for key in ("content", "created_by"):
        if not isinstance(value.get(key), str):
            raise ValueError(f"body compaction requires {key}")
    _strings(value, ("created_at",))
    # v2's uncertain single-join form is parsed only for explicit recovery.
    if "source_uids" in value:
        fields = ("source_uids",)
    else:
        if value.get("version") != 1 or type(value.get("version")) is not int:
            raise ValueError("unsupported body compaction version")
        fields = ("derivation_source_uids", "restore_group_uids")
    for key in fields:
        sources = value.get(key)
        _string_list(sources, key)
        if not sources or len(sources) != len(set(sources)):
            raise ValueError(f"{key} must be nonempty and unique")


def validate_message_record(d: Any) -> None:
    _object(d, "message record")
    if (not isinstance(d.get("uid"), str) or not d["uid"]
            or not isinstance(d.get("role"), str) or not d["role"]
            or not isinstance(d.get("content"), str)):
        raise ValueError("message requires UID, role and indexing content")
    _strings(d, ("uid", "role", "content", "source", "created_at", "toolobs_ref"),
             nullable=("toolobs_ref",))
    _integers(d, ("index", "token_count", "toolobs_chars"),
              nullable=("token_count", "toolobs_chars"))
    _booleans(d, ("protected", "active", "archived"))
    for key in ("summary_parent_ids", "permanent_reasons", "temporary_reasons"):
        _string_list(d.get(key, []), key)
    metadata = d.get("metadata", {})
    _object(metadata, "metadata")
    if "tool_call_ids" in metadata:
        _string_list(metadata["tool_call_ids"], "tool_call_ids")
    _booleans(metadata, ("clm_pin",))
    _strings(metadata, ("protect_reason", "tool_name"))
    envelope = d.get("envelope")
    if envelope is not None:
        _object(envelope, "canonical envelope")
        validate_call_container(envelope)
        if not isinstance(envelope.get("role"), str) or not envelope["role"]:
            raise ValueError("canonical envelope requires a role")
        # Provider extensions remain arbitrary JSON. They are not record metadata.
        try:
            json.dumps(envelope, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("canonical envelope must be JSON") from exc
    if d.get("body_compaction") is not None:
        validate_body_compaction(d["body_compaction"])
    history = d.get("body_compaction_history", [])
    if not isinstance(history, list):
        raise ValueError("body compaction history must be a list")
    for overlay in history:
        validate_body_compaction(overlay)


def validate_summary_record(d: Any) -> None:
    _object(d, "summary record")
    if (not isinstance(d.get("uid"), str) or not d["uid"]
            or not isinstance(d.get("content"), str)):
        raise ValueError("summary requires UID and content")
    _strings(d, ("uid", "content", "created_by", "created_at"))
    _string_list(d.get("source_message_ids", []), "source_message_ids")
    _integers(d, ("revision", "anchor_index", "token_count"), nullable=("token_count",))
    _booleans(d, ("active",))
    if "version" in d or "derivation_source_uids" in d:
        if type(d.get("version")) is not int or d["version"] != 1:
            raise ValueError("invalid narrative version")
        for key in ("source_message_ids", "derivation_source_uids"):
            values = d.get(key)
            _string_list(values, key)
            if not values or len(values) != len(set(values)):
                raise ValueError(f"{key} must be nonempty and unique")
        if not set(d["source_message_ids"]).issubset(d["derivation_source_uids"]):
            raise ValueError("narrative membership outside derivation")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


@dataclass
class ContextMessage:
    """A record for one canonical conversation message (spec §6.1).

    ``uid`` fingerprints the position and complete provider envelope (v2).
    ``content`` is indexing text; ``envelope`` is the original typed payload.
    Envelope getters and serialized snapshots return detached nested values.
    """

    uid: str
    role: str
    content: str
    source: str = "canonical"
    created_at: str = ""
    token_count: Optional[int] = None
    protected: bool = False
    active: bool = True
    archived: bool = False
    summary_parent_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    index: int = 0
    # Tool-output offloading (spec §27): full text stored externally.
    toolobs_ref: Optional[str] = None
    toolobs_chars: Optional[int] = None

    # Serialized privately so even a getter's nested values cannot alias it.
    _envelope_json: Optional[str] = field(default=None, repr=False)
    body_compaction: Optional[Dict[str, Any]] = None
    body_compaction_history: List[Dict[str, Any]] = field(default_factory=list)
    permanent_reasons: List[str] = field(default_factory=list)
    temporary_reasons: List[str] = field(default_factory=list)

    @property
    def envelope(self) -> Optional[Dict[str, Any]]:
        return json.loads(self._envelope_json) if self._envelope_json is not None else None

    def set_envelope(self, envelope: Dict[str, Any]) -> None:
        _object(envelope, "canonical envelope")
        validate_call_container(envelope)
        if not isinstance(envelope.get("role"), str) or not envelope["role"]:
            raise ValueError("canonical envelope requires a role")
        self._envelope_json = json.dumps(envelope, ensure_ascii=False, allow_nan=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uid": self.uid,
            "role": self.role,
            "content": self.content,
            "source": self.source,
            "created_at": self.created_at,
            "token_count": self.token_count,
            "protected": self.protected,
            "active": self.active,
            "archived": self.archived,
            "summary_parent_ids": list(self.summary_parent_ids),
            "metadata": deepcopy(self.metadata),
            "index": self.index,
            "toolobs_ref": self.toolobs_ref,
            "toolobs_chars": self.toolobs_chars,
            "envelope": self.envelope,
            "body_compaction": deepcopy(self.body_compaction),
            "body_compaction_history": deepcopy(self.body_compaction_history),
            "permanent_reasons": list(self.permanent_reasons),
            "temporary_reasons": list(self.temporary_reasons),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ContextMessage":
        validate_message_record(d)
        return cls(
            uid=d.get("uid", ""),
            role=d.get("role", ""),
            content=d.get("content", ""),
            source=d.get("source", "canonical"),
            created_at=d.get("created_at", ""),
            token_count=d.get("token_count"),
            protected=d.get("protected", False),
            active=d.get("active", True),
            archived=d.get("archived", False),
            summary_parent_ids=list(d.get("summary_parent_ids", [])),
            metadata=deepcopy(d.get("metadata", {})),
            index=d.get("index", 0),
            toolobs_ref=d.get("toolobs_ref"),
            toolobs_chars=d.get("toolobs_chars"),
            _envelope_json=(json.dumps(d["envelope"], ensure_ascii=False, allow_nan=False)
                            if d.get("envelope") is not None else None),
            body_compaction=deepcopy(d.get("body_compaction")),
            body_compaction_history=deepcopy(d.get("body_compaction_history", [])),
            permanent_reasons=list(d.get("permanent_reasons", [])),
            temporary_reasons=list(d.get("temporary_reasons", [])),
        )


@dataclass
class ContextSummary:
    """Synthetic summary record (spec §6.2). Originals are never destroyed."""

    uid: str
    content: str
    source_message_ids: List[str] = field(default_factory=list)
    created_by: str = "model"
    created_at: str = ""
    token_count: Optional[int] = None
    revision: int = 1
    # Position of the first source message: the summary renders at this index.
    anchor_index: int = 0
    active: bool = True
    # None is the independently parsable legacy form, never accepted v4 lineage.
    version: Optional[int] = None
    derivation_source_uids: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        record = {
            "uid": self.uid,
            "content": self.content,
            "source_message_ids": list(self.source_message_ids),
            "created_by": self.created_by,
            "created_at": self.created_at,
            "token_count": self.token_count,
            "revision": self.revision,
            "anchor_index": self.anchor_index,
            "active": self.active,
        }
        if self.version is not None or self.derivation_source_uids is not None:
            record["version"] = self.version
            record["derivation_source_uids"] = deepcopy(self.derivation_source_uids)
        return record

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ContextSummary":
        validate_summary_record(d)
        return cls(
            uid=d.get("uid", ""),
            content=d.get("content", ""),
            source_message_ids=list(d.get("source_message_ids", [])),
            created_by=d.get("created_by", "model"),
            created_at=d.get("created_at", ""),
            token_count=d.get("token_count"),
            revision=d.get("revision", 1),
            anchor_index=d.get("anchor_index", 0),
            active=d.get("active", True),
            version=d.get("version"),
            derivation_source_uids=deepcopy(d.get("derivation_source_uids")),
        )


@dataclass
class ContextState:
    """Explicit CLM working-context state (spec §6.3)."""

    conversation_id: str = ""
    session_id: str = ""
    active_items: List[str] = field(default_factory=list)   # ordered uids in working set
    archived_items: List[str] = field(default_factory=list)  # uids outside working set
    summaries: Dict[str, ContextSummary] = field(default_factory=dict)
    current_tokens: int = 0
    budget_tokens: int = 0
    reserve_tokens: int = 4096
    last_compaction_at: Optional[str] = None
    last_compaction_tokens: Optional[int] = None
    context_revision: int = 0
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "session_id": self.session_id,
            "active_items": list(self.active_items),
            "archived_items": list(self.archived_items),
            "summaries": {k: v.to_dict() for k, v in self.summaries.items()},
            "current_tokens": self.current_tokens,
            "budget_tokens": self.budget_tokens,
            "reserve_tokens": self.reserve_tokens,
            "last_compaction_at": self.last_compaction_at,
            "last_compaction_tokens": self.last_compaction_tokens,
            "context_revision": self.context_revision,
            "metrics": dict(self.metrics),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ContextState":
        _object(d, "context state")
        _strings(d, ("conversation_id", "session_id", "last_compaction_at"),
                 nullable=("last_compaction_at",))
        for key in ("active_items", "archived_items"):
            _string_list(d.get(key, []), key)
        _object(d.get("summaries", {}), "summaries")
        validate_metrics(d.get("metrics", {}))
        _integers(d, ("current_tokens", "budget_tokens", "reserve_tokens",
                      "context_revision", "last_compaction_tokens"),
                  nullable=("last_compaction_tokens",))
        return cls(
            conversation_id=d.get("conversation_id", ""),
            session_id=d.get("session_id", ""),
            active_items=list(d.get("active_items", [])),
            archived_items=list(d.get("archived_items", [])),
            summaries={
                k: ContextSummary.from_dict(v) for k, v in d.get("summaries", {}).items()
            },
            current_tokens=d.get("current_tokens", 0),
            budget_tokens=d.get("budget_tokens", 0),
            reserve_tokens=d.get("reserve_tokens", 4096),
            last_compaction_at=d.get("last_compaction_at"),
            last_compaction_tokens=d.get("last_compaction_tokens"),
            context_revision=d.get("context_revision", 0),
            metrics=dict(d.get("metrics", {})),
        )


@dataclass(frozen=True)
class ProtectionSnapshot:
    """Detached explanation of protection on one current canonical branch."""
    protected_uids: tuple[str, ...]
    permanent_reasons: Dict[str, tuple[str, ...]]
    temporary_reasons: Dict[str, tuple[str, ...]]
