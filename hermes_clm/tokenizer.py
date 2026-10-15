"""Token counting with a conservative fallback hierarchy (spec §34).

Hierarchy:
  1. exact model tokenizer (tiktoken, if available and configured)
  2. provider estimator — the Hermes host counts real usage and feeds it back
     via ``update_from_response``; the engine recalibrates its heuristic
     against real counts so budget decisions track the provider.
  3. configured generic tokenizer (``tokenizer: "tiktoken:<enc>"``)
  4. character-based conservative estimate (chars / 3.5, rounded up)

The controller errs on the high side: the heuristic ratio is *maxed* across
calibration samples, and uncounted content is never estimated at zero.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

_HEUR_CHARS_PER_TOKEN = 3.5
_CALIBRATION_MIN_CHARS = 512
_MAX_RATIO = 1.6  # sanity cap on the calibrated chars/token ratio


class TokenCounter:
    def __init__(self, mode: str = "auto"):
        self.mode = mode or "auto"
        self._enc = None
        self._enc_name: Optional[str] = None
        self._ratio: Optional[float] = None  # calibrated chars/token
        self._samples: List[float] = []
        if self.mode.startswith("tiktoken:"):
            self._enc_name = self.mode.split(":", 1)[1]
            self._load_tiktoken(self._enc_name)
        elif self.mode == "auto":
            # Try the common encoding silently; heuristic remains the fallback.
            self._load_tiktoken("o200k_base")

    def _load_tiktoken(self, name: str) -> None:
        try:
            import tiktoken  # type: ignore

            self._enc = tiktoken.get_encoding(name)
            self._enc_name = name
        except Exception:
            self._enc = None
            self._enc_name = None

    @property
    def backend_name(self) -> str:
        return f"tiktoken:{self._enc_name}" if self._enc else "heuristic"

    # --- counting ---------------------------------------------------------

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        if self._enc is not None:
            try:
                return len(self._enc.encode(text))
            except Exception:
                pass
        return self._heuristic_count(text)

    def _heuristic_count(self, text: str) -> int:
        if self._ratio:
            return max(1, math.ceil(len(text) / self._ratio))
        return max(1, math.ceil(len(text) / _HEUR_CHARS_PER_TOKEN))

    def count_message(self, message: Dict[str, Any]) -> int:
        """Estimate tokens for one OpenAI-format message (content + tool names)."""
        tokens = 4  # per-message overhead
        content = message.get("content")
        if isinstance(content, str):
            tokens += self.count_text(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text" and part.get("text"):
                        tokens += self.count_text(str(part["text"]))
                    elif part.get("type") == "tool_use" and part.get("id"):
                        tokens += self.count_text(str(part["id"]))
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                fn = call.get("function") or {}
                tokens += self.count_text(str(fn.get("name", "")))
                tokens += self.count_text(str(fn.get("arguments", "")))
        return tokens

    def count_messages(self, messages: List[Dict[str, Any]]) -> int:
        return sum(self.count_message(m) for m in messages if isinstance(m, dict))

    # --- calibration against real provider usage (spec §34, err high) ------

    def calibrate(self, estimated_tokens: int, real_prompt_tokens: int,
                  estimated_chars: int, real_chars: int) -> None:
        """Feed a (estimate, real) pair; tighten the chars/token ratio high-side.

        Only samples with enough text are used; the ratio is the *maximum*
        observed (conservative), which biases budget decisions upward.
        """
        if estimated_chars < _CALIBRATION_MIN_CHARS or real_prompt_tokens < 100:
            return
        # The prompt includes more than our counted messages (system prompt,
        # tool schemas, notices) — cap the ratio so those extras don't make
        # the heuristic absurdly aggressive.
        ratio = estimated_chars / float(real_prompt_tokens)
        if ratio <= 0 or ratio > _MAX_RATIO:
            return
        self._samples.append(ratio)
        self._samples = self._samples[-64:]
        self._ratio = max(self._ratio or 0.0, max(self._samples))
