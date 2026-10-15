# hermes-clm — Context Language Models for Hermes Agent

Model-directed, **reversible** working-context management for Hermes Agent.
The model decides what stays in its active context, what gets condensed into
summaries, and what gets archived (and later restored) — while the canonical
Hermes conversation remains untouched underneath.

> **Licensing note (spec §51):** this project is *inspired by* Context
> Language Models (CLM) and is an **independent** implementation written for
> Hermes Agent. Paper: **Context Language Models**
> ([arXiv:2609.37725](https://arxiv.org/abs/2609.37725), Shao et al., 2026 —
> UW + Meta Superintelligence Labs + MIT); official code:
> [facebookresearch/context-language-models](https://github.com/facebookresearch/context-language-models).
> Meta's CLM reference implementation is **not** bundled. Licensed MIT.

---

## What it does

```text
Canonical Hermes History          (immutable, append-only — sacred)
        │
        ▼
CLM Context Engine State
  ├── active working set          (what the model actually sees)
  ├── synthetic summaries         (model-written surrogates)
  ├── archived spans              (restorable, never destroyed)
  ├── decision ledger / telemetry (events.jsonl)
  └── context metadata            (revision, budget, provenance)
        │
        ▼
select_context()  →  model inference  →  clm_* tools  →  next turn
```

Instead of Hermes' fixed truncation / external summarizer, the *model* runs a
small tool surface (`clm_status`, `clm_compact`, `clm_archive`, `clm_restore`,
`clm_replace_summary`) to manage its own context. A deterministic budget
controller and policy layer keep it honest: protected regions, hard budgets,
reversibility, and a host fallback if the model mismanages context.

## Core guarantees

- **Canonical history is sacred** — never forked, modified, or destroyed.
  `select_context()` returns a cloned, request-only list.
- **Everything is reversible** — a "drop" means *removed from the active set*,
  not erased. Archives stay restorable by stable UID.
- **Protected prefix** — the system prompt, root task, and recent turns can
  never be compacted/archived by the model.
- **Budget is internal** — CLM targets a small *working set* (default 32K)
  while the model/provider still advertises the real ≥ 64K window. The 32K
  target is never reported back as the model context length.
- **Fail-safe** — if the model over/under-compresses, a deterministic host
  fallback preserves the session. The agent never crashes on context failure.

## Install

The engine package is pure Python (stdlib `json` only). Install into the
Hermes venv:

```bash
cd /path/to/hermes-clm
pip install -e .              # optional local tokenizer: pip install -e ".[tiktoken]"
```

Installing tiktoken alone does not qualify complete Qwen/provider request budgets.
Exact mode requires the explicitly bound provider/tokenizer/template service and
parity evidence described in [exact-budget.md](docs/exact-budget.md).

The plugin entry point lives in `~/.hermes/plugins/hermes-clm/` (a thin shim
over the package). Hermes discovers it automatically.

## Enable

In `~/.hermes/config.yaml`:

```yaml
context:
  engine: hermes-clm
```

(Keep `model.context_length` at the provider's real value, ≥ 64K. The CLM
working target is separate.)

### Configuration

```yaml
clm:
  active_target_tokens: 32768     # internal working-set target
  reserve_tokens: 4096            # reserved for output/tools/notices
  provider_context_min_tokens: 64000   # Hermes hard floor (never lowered)
  nudge_thresholds: [0.25, 0.50, 0.75, 0.90]
  protect_latest_user_turns: 2
  protect_latest_assistant_turns: 1
  allow_archive: true
  allow_compaction: true
  allow_restore: true
  allow_summary_revision: true
  overflow_compaction_retries: 6
  emergency_host_fallback: true
  persistence: true               # state.json per conversation
  event_log: true                 # events.jsonl per conversation
  tool_output_soft_limit_chars: 12000
  tool_output_hard_limit_chars: 50000
```

## Inspect (spec §48/§49)

```bash
/clm status              # working budget, utilization, counts, revision
/clm active              # the active working set
/clm archive             # what's archived (restorable)
/clm summary <id>        # a synthetic summary + provenance
/clm events              # recent context-operation events
/clm restore <id>        # restore a summary's sources (budget-aware)
/clm reset-working-set   # rebuild active set from canonical history
```

## Tool surface (spec §9)

| Tool | Purpose |
|------|---------|
| `clm_status` | Inspect budget, utilization, active/archived/summaries, headroom |
| `clm_compact` | Replace a set/range of active messages with a model-written summary |
| `clm_archive` | Remove from active set **without** a summary (still restorable) |
| `clm_restore` | Bring archived material back (budget-aware) |
| `clm_replace_summary` | Improve/correct an existing synthetic summary |

## State & telemetry (spec §29/§30)

Per-conversation, under `~/.hermes/clm/<conversation_id>/`:

- `state.json` — full CLM state (messages, summaries, archive, revision)
- `events.jsonl` — append-only operation log (initiator, tokens, revisions)

## Development

```bash
pip install -e ".[dev]"
pytest                     # unit tests (tests/)
```

Module layout (spec §38):

```text
hermes_clm/
├── engine.py        # CLMEngine — host hook wiring
├── config.py        # CLMConfig + resolution
├── models.py        # ContextMessage / ContextSummary / ContextState
├── store.py         # ContextStore — canonical CLM state, persistence
├── selector.py      # select_context ordering
├── budget.py        # ContextBudgetController
├── policies.py      # protected regions, validation, emergency order
├── prompts.py       # model-facing guidance + toolobs placeholder
├── telemetry.py     # event log
├── tokenizer.py     # token counting (tiktoken → provider → char fallback)
└── tools/           # the five clm_* tools + dispatch
```

See `docs/` for architecture, benchmarking, context policy, and
troubleshooting.
