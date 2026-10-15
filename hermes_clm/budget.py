"""Budget controller (spec §16, §40).

Owns token estimates, the usable budget, the reserve, threshold state, overflow
detection, and compaction retry counts. It does NOT decide what information to
remove — that stays the model's job (host fallback is the exception of last
resort, see policies.py).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


class ContextBudgetController:
    def __init__(self, config: Any, provider_context_length: int = 0):
        self.config = config
        self.provider_context_length = max(int(provider_context_length or 0),
                                           config.provider_context_min_tokens)
        self.active_target = config.active_target_tokens
        self.reserve = config.reserve_tokens
        self._nudges_issued: List[float] = []
        self._overflow_retries = 0
        self.peak_active_tokens = 0
        self._active_samples: List[int] = []

    # --- derived budgets ---------------------------------------------------

    @property
    def usable_tokens(self) -> int:
        """Normal working budget: target minus reserve (spec §16)."""
        return max(1024, self.active_target - self.reserve)

    @property
    def hard_limit(self) -> int:
        """Never exceed this in a selected request.

        min(provider ceiling * overflow_fraction, active_target) — the provider
        window stays the safety headroom, the CLM target the operating point.
        """
        provider_cap = int(self.provider_context_length * self.config.overflow_fraction_of_provider)
        return min(self.active_target, provider_cap)

    # --- estimation --------------------------------------------------------

    def estimate(self, store: Any) -> int:
        tokens = store.working_set_tokens()
        self.observe(tokens)
        return tokens

    def observe(self, active_tokens: int) -> None:
        """Track peak/mean/median (spec §31)."""
        active_tokens = max(0, int(active_tokens))
        self.peak_active_tokens = max(self.peak_active_tokens, active_tokens)
        self._active_samples.append(active_tokens)
        self._active_samples = self._active_samples[-1024:]

    def headroom(self, store: Any) -> int:
        """Free space in the working budget (used by restore error reporting)."""
        return max(0, self.usable_tokens - store.working_set_tokens())

    def metrics_snapshot(self) -> Dict[str, Any]:
        return {
            "peak_active_context_tokens": self.peak_active_tokens,
            "mean_active_context_tokens": round(sum(self._active_samples) / len(self._active_samples), 1)
            if self._active_samples else 0,
            "median_active_context_tokens": (
                sorted(self._active_samples)[len(self._active_samples) // 2]
                if self._active_samples else 0
            ),
        }

    # --- nudges (spec §17) --------------------------------------------------

    def utilization(self, active_tokens: int) -> float:
        return active_tokens / self.usable_tokens if self.usable_tokens else 0.0

    def nudge_level(self, active_tokens: int) -> Optional[float]:
        """Highest threshold crossed, or None below the first threshold.

        Only issues a nudge for a threshold NOT already issued at this session's
        level; the host injects the notice (prompts.py).
        """
        util = self.utilization(active_tokens)
        crossed = [t for t in self.config.nudge_thresholds if util >= t]
        if not crossed:
            return None
        level = max(crossed)
        if level in self._nudges_issued:
            return None
        self._nudges_issued.append(level)
        self._nudges_issued.sort()
        return level

    def reset_nudges(self) -> None:
        self._nudges_issued = []

    # --- overflow (spec §18) -------------------------------------------------

    def will_overflow(self, active_tokens: int, expansion_tokens: int) -> bool:
        return active_tokens + expansion_tokens > self.hard_limit

    def begin_overflow_retry(self) -> int:
        self._overflow_retries += 1
        return self._overflow_retries

    def overflow_retries_exhausted(self) -> bool:
        return self._overflow_retries > self.config.overflow_compaction_retries

    def reset_overflow_retries(self) -> None:
        self._overflow_retries = 0

    # --- status (spec §10 response shape) -------------------------------------

    def status(self, store: Any, active_tokens: int) -> Dict[str, Any]:
        return {
            "budget_tokens": self.usable_tokens,
            "reserve_tokens": self.reserve,
            "active_tokens": active_tokens,
            "utilization": round(self.utilization(active_tokens), 4),
            "active_messages": len(store.active_messages()),
            "archived_messages": len(store.archived_messages()),
            "summaries": len(store.summaries),
            "headroom_tokens": max(0, self.usable_tokens - active_tokens),
            "context_revision": store.revision,
            "provider_context_length": self.provider_context_length,
            "hard_limit_tokens": self.hard_limit,
            "overflow_retries": self._overflow_retries,
        }
