"""CLM configuration (spec §16, §36).

Hermes' hard rule: the model/provider context window must advertise >= 64K
tokens. CLM's ``active_target_tokens`` is an *internal working-set target*,
never something reported back as the model context length.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

PROVIDER_CONTEXT_MIN_TOKENS = 64_000


def _env(name: str, default: Any) -> Any:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    if isinstance(default, bool):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(default, int) and not isinstance(default, bool):
        try:
            return int(value)
        except ValueError:
            return default
    if isinstance(default, float):
        try:
            return float(value)
        except ValueError:
            return default
    if isinstance(default, list):
        try:
            parts = [p.strip() for p in str(value).replace(";", ",").split(",") if p.strip()]
            return parts
        except Exception:
            return default
    return value


@dataclass
class CLMConfig:
    """Engine configuration with spec §36 defaults."""

    # --- budget (spec §16) ---
    active_target_tokens: int = 32_768
    reserve_tokens: int = 4_096
    provider_context_min_tokens: int = PROVIDER_CONTEXT_MIN_TOKENS
    budget_mode: str = "approximate"  # exact requires an explicitly bound provider counter
    output_tokens: int = 4096
    safety_margin_tokens: int = 1024

    # --- nudging (spec §17) ---
    nudge_thresholds: List[float] = field(
        default_factory=lambda: [0.25, 0.50, 0.75, 0.90])

    # --- protection (spec §8) ---
    protect_latest_user_turns: int = 2
    protect_latest_assistant_turns: int = 1

    # --- tool enablement (spec §9) ---
    allow_status: bool = True
    allow_compaction: bool = True
    allow_archive: bool = True
    allow_restore: bool = True
    allow_summary_revision: bool = True

    # --- overflow handling (spec §18) ---
    overflow_compaction_retries: int = 6
    emergency_host_fallback: bool = True
    # Bounded recovery turn (recovery-design option 2, docs/plans/): host-
    # triggered initiation, one model-owned selection turn per pressure
    # episode, deterministic fail-closed fallback. Off by default — turning
    # it on is a reviewed activation decision, not an implementation default.
    recovery_turn: bool = False
    # Fraction of the *hard* provider ceiling the engine treats as overflow
    # for the request it selected (safety headroom, spec §16).
    overflow_fraction_of_provider: float = 0.95

    # --- persistence (spec §29) ---
    persistence: bool = True
    event_log: bool = True
    state_root: str = ""  # default: <hermes_home>/clm

    # --- tool-output offloading (spec §27) ---
    tool_output_soft_limit_chars: int = 12_000
    tool_output_hard_limit_chars: int = 50_000
    tool_output_window_chars: int = 4_000

    # --- tokenizer (spec §34) ---
    tokenizer: str = "auto"  # auto | tiktoken:<encoding> | heuristic

    # --- misc ---
    max_restoration_tokens: int = 0  # 0 = no extra cap beyond hard budget
    emit_nudges: bool = True

    # Map state is experimental and never changes baseline plugin selection.
    context_map_enabled: bool = False
    context_map_token_cap: int = 2000
    context_map_token_target: int = 1500
    context_map_max_depth: int = 5
    context_map_description_limit: int = 256
    context_map_max_results: int = 5
    context_map_mode: str = 'deterministic'
    context_map_browse: bool = False
    # Legacy retrofit adoption: a store that predates the map can contain
    # branch-external superseded versions ("ghosts") that can never be
    # witnessed or audited, so coverage refuses forever. When explicitly
    # enabled, such records are retired from the live state and preserved
    # byte-for-byte in a sidecar next to state.json instead of wedging the
    # session. Off by default — adoption of old history is an explicit
    # choice, not an implementation default.
    retrofit_retire_ghosts: bool = False

    def __post_init__(self) -> None:
        self.validate_context_map()

    @property
    def usable_target_tokens(self) -> int:
        """Working budget the model is expected to stay under (target minus reserve)."""
        return max(1024, self.active_target_tokens - self.reserve_tokens)

    @classmethod
    def from_hermes(cls, agent_cfg: Optional[Dict[str, Any]] = None,
                    hermes_home: str = "") -> "CLMConfig":
        """Build config from Hermes ``context:`` / ``clm:`` config sections plus
        ``HERMES_CLM_*`` env overrides."""
        cfg = cls()
        if hermes_home:
            cfg.state_root = f"{hermes_home}/clm"
        if isinstance(agent_cfg, dict):
            ctx = agent_cfg.get("context", {})
            clm = agent_cfg.get("clm", {})
            ctx_clm = ctx.get("clm", {}) if isinstance(ctx, dict) else {}
            merged: Dict[str, Any] = {}
            if isinstance(ctx_clm, dict):
                merged.update(ctx_clm)
            if isinstance(clm, dict):
                merged.update(clm)
            for key in (
                "active_target_tokens", "reserve_tokens", "nudge_thresholds",
                "protect_latest_user_turns", "protect_latest_assistant_turns",
                "allow_status", "allow_compaction", "allow_archive", "allow_restore",
                "allow_summary_revision", "overflow_compaction_retries",
                "emergency_host_fallback", "recovery_turn", "persistence", "event_log",
                "state_root", "tool_output_soft_limit_chars",
                "tool_output_hard_limit_chars", "tool_output_window_chars",
                "tokenizer", "max_restoration_tokens", "emit_nudges",
                "budget_mode", "output_tokens", "safety_margin_tokens",
                "context_map_enabled", "context_map_token_cap", "context_map_token_target",
                "context_map_max_depth", "context_map_description_limit", "context_map_max_results",
                "context_map_mode", "context_map_browse", "retrofit_retire_ghosts",
            ):
                if key in merged and merged[key] is not None:
                    setattr(cfg, key, merged[key])
            if isinstance(ctx, dict) and isinstance(ctx.get("nudge_thresholds"), list):
                cfg.nudge_thresholds = list(ctx["nudge_thresholds"])
        # Env overrides (useful for experiments without config edits).
        overrides = {
            "HERMES_CLM_ACTIVE_TARGET": "active_target_tokens",
            "HERMES_CLM_RESERVE": "reserve_tokens",
            "HERMES_CLM_PERSISTENCE": "persistence",
            "HERMES_CLM_EVENT_LOG": "event_log",
            "HERMES_CLM_STATE_ROOT": "state_root",
            "HERMES_CLM_TOOLKIT": "tokenizer",
            "HERMES_CLM_RETROFIT_RETIRE_GHOSTS": "retrofit_retire_ghosts",
        }
        for env_name, attr in overrides.items():
            value = _env(env_name, None)
            if value is not None:
                setattr(cfg, attr, value)
        thresholds = _env("HERMES_CLM_NUDGE_THRESHOLDS", None)
        if thresholds:
            cfg.nudge_thresholds = [float(t) for t in thresholds]
        cfg._validate()
        return cfg

    def validate_retention(self) -> None:
        for name in ("protect_latest_user_turns", "protect_latest_assistant_turns"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")

    def validate_budget(self) -> None:
        if self.budget_mode not in {"approximate", "exact"}:
            raise ValueError("budget_mode must be approximate or exact")
        for name, minimum in (("output_tokens", 1), ("safety_margin_tokens", 0)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.budget_mode == "exact":
            if type(self.max_restoration_tokens) is not int or self.max_restoration_tokens < 0:
                raise ValueError("max_restoration_tokens must be a nonnegative integer")
            from .provider_budget import BudgetLimits
            BudgetLimits(active_target=self.active_target_tokens, reserve=self.reserve_tokens,
                         output_tokens=self.output_tokens, safety_margin=self.safety_margin_tokens)

    def validate_context_map(self) -> None:
        for name in ('context_map_enabled', 'context_map_browse'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f'{name} must be a boolean')
        for name, lower, upper in (
            ('context_map_token_cap', 1, 2000),
            ('context_map_token_target', 1, 2000),
            ('context_map_max_depth', 1, 5),
            ('context_map_description_limit', 0, 256),
            ('context_map_max_results', 1, 5),
        ):
            value = getattr(self, name)
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f'{name} must be an integer between {lower} and {upper}')
        if self.context_map_token_target > self.context_map_token_cap:
            raise ValueError('context_map_token_target must not exceed context_map_token_cap')
        if type(self.context_map_mode) is not str or self.context_map_mode != 'deterministic':
            raise ValueError('context_map_mode must be deterministic')
        if self.context_map_browse:
            raise ValueError('context_map_browse requires separately verified navigation support')

    def _validate(self) -> None:
        self.validate_retention()
        self.validate_budget()
        self.validate_context_map()
        if self.active_target_tokens < 2048:
            raise ValueError("active_target_tokens must be >= 2048")
        if self.reserve_tokens < 0:
            raise ValueError("reserve_tokens must be >= 0")
        if self.active_target_tokens > self.provider_context_min_tokens:
            # Not fatal: CLM may target up to the real provider ceiling, but it
            # must never be below Hermes' 64K floor requirement — and if it is
            # above the floor, the *provider* still advertises its own (larger)
            # window. We only warn for misconfiguration below the floor.
            pass
        if not all(0 < t < 1 for t in self.nudge_thresholds):
            raise ValueError("nudge_thresholds must be fractions in (0, 1)")
        if self.overflow_compaction_retries < 0:
            raise ValueError("overflow_compaction_retries must be >= 0")
        if type(self.recovery_turn) is not bool:
            raise ValueError("recovery_turn must be a boolean")
        if type(self.retrofit_retire_ghosts) is not bool:
            raise ValueError("retrofit_retire_ghosts must be a boolean")


def resolve_state_root(config: CLMConfig, hermes_home: str = "") -> str:
    """<hermes_home>/clm by default (spec §29)."""
    if config.state_root:
        return config.state_root
    if hermes_home:
        return f"{hermes_home}/clm"
    home = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
    return f"{home}/clm"
