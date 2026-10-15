# Changelog

## 0.1.0 — 2026-10-01

First usable release (spec v0.2 §53 acceptance criteria).

### Added
- `CLMEngine` context-engine plugin implementing the Hermes `ContextEngine`
  contract (`select_context`, `on_turn_complete`, `get_tool_schemas`,
  `handle_tool_call`, `update_from_response`, `clone_for_agent`,
  `on_session_start/end/reset`).
- Five model-facing CLM tools (spec §9): `clm_status`, `clm_compact`,
  `clm_archive`, `clm_restore`, `clm_replace_summary`.
- `ContextStore` with deterministic message UIDs, idempotent ingestion,
  protected system slot, reversible archive, and provenance-carrying
  synthetic summaries.
- `ContextBudgetController`: 32K internal working-set target (never reported
  as model context length), 4K reserve, nudge thresholds, overflow retry
  counter, peak/mean/median metrics.
- Policy/validation layer: protected-context rejection, unknown-ID rejection,
  budget-aware restore with required-token reporting, atomic
  snapshot/rollback on invalid operations.
- Emergency host fallback: overflow trim + deterministic emergency compaction,
  clearly labeled in telemetry.
- Tool-output offloading (`toolobs_*` artifacts) with head/tail windows.
- JSON persistence (`state.json`) + append-only event log (`events.jsonl`)
  per conversation under `~/.hermes/clm/<conversation_id>/`.
- `/clm` inspection command surface (status/active/archive/summary/events/
  restore/reset-working-set) via the Hermes plugin shim.
- Test suite: 37 unit tests covering store invariants, budget, compaction
  atomicity, protected context, reversible restore, persistence round-trip,
  and overflow behavior.

### Notes
- Independent implementation inspired by Context Language Models
  (Shao et al., 2026, arXiv:2609.37725); Meta's reference implementation
  is not bundled (spec §51).
- Pure stdlib; optional `[tiktoken]` extra for exact token counts.
