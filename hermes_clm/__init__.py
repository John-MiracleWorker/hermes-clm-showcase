"""hermes-clm: a model-directed, reversible context-management engine.

Hermes Agent context-engine plugin. The model operates on a structured
working-context layer (compact / archive / restore / replace-summary) while
the canonical Hermes conversation underneath stays immutable (spec §60).

Independent implementation inspired by Context Language Models (CLM;
Shao et al., 2026, arXiv:2609.37725).
"""

from __future__ import annotations

from .config import CLMConfig, PROVIDER_CONTEXT_MIN_TOKENS, resolve_state_root
from .engine import CLMEngine
from .models import ContextMessage, ContextState, ContextSummary, utc_now_iso
from .retrieval import ContextMapIndex, ContextMapNode
from .store import ContextStore, message_uid, summary_uid, toolobs_uid
from .tokenizer import TokenCounter

__version__ = "0.1.0"

__all__ = [
    "CLMEngine",
    "CLMConfig",
    "PROVIDER_CONTEXT_MIN_TOKENS",
    "resolve_state_root",
    "ContextStore",
    "ContextMessage",
    "ContextState",
    "ContextSummary",
    "ContextMapIndex",
    "ContextMapNode",
    "TokenCounter",
    "message_uid",
    "summary_uid",
    "toolobs_uid",
    "utc_now_iso",
    "__version__",
]
