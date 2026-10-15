# Architecture

`hermes-clm` plugs into Hermes Agent through the `ContextEngine` contract.
It is a **thin plugin** over the `hermes_clm` package: Hermes owns the
canonical conversation, CLM owns only the *effective working context* that
each inference call sees.

## Data flow

```text
Hermes conversation (canonical, append-only, SACRED)
        │  completed turns
        ▼
CLMEngine.on_turn_complete()
        │
        ▼
ContextStore
   ├── system slot          (protected, always current)
   ├── messages {}          (canonical mirror, stable UIDs)
   ├── order []             (active ordering)
   ├── summaries {}         (model-written surrogates + provenance)
   └── toolobs {}           (offloaded large tool outputs)
        │
        ▼
CLMEngine.select_context(request_messages)
   1. load protected prefix + CLM guidance
   2. load active raw messages
   3. load active summaries at their logical positions
   4. inject context notice if a nudge threshold is crossed
   5. enforce hard budget (trim / emergency compaction)
   6. return a CLONED, request-only message list
        │
        ▼
Model inference  ──tool calls──▶  clm_* tools  ──▶  store mutation
```

**Key invariant:** `select_context()` never mutates canonical history. It
returns a fresh list. The store is the only writer of CLM state, and it never
invokes the model (spec §39).

## Modules

| Module | Responsibility (spec §38) |
|--------|---------------------------|
| `engine.py` | `CLMEngine` — host hook wiring, orchestration, fail-open guards |
| `store.py` | canonical CLM state, ordering, archive, summaries, provenance, serialization |
| `budget.py` | token estimates, usable budget, reserve, thresholds, overflow, retry counts |
| `policies.py` | protected regions, recent-turn retention, validation, emergency order |
| `selector.py` | `select_context()` ordering (spec §20) |
| `tokenizer.py` | tiktoken → provider estimator → char fallback (spec §34) |
| `prompts.py` | model-facing guidance + toolobs placeholder text (spec §15) |
| `telemetry.py` | `events.jsonl` writer + metrics snapshot |
| `config.py` | `CLMConfig` + Hermes config resolution |
| `tools/` | the five `clm_*` tools + `dispatch()` error boundary |

## UIDs and determinism

Message UIDs are derived deterministically from `(index, role, content_hash)`:
re-ingesting the same conversation is idempotent, restarts converge, and no
counter state is needed (spec §28). Summaries get `clm_summary_NNNN`,
offloaded outputs `toolobs_NNNNN`.

## The 64K floor

Hermes requires a model/provider context of **≥ 64K tokens**. CLM's
`active_target_tokens` (default 32K) is an *internal working-set target* — it
is **never** reported back as the model context length. The provider's real
window (64K, 131K, …) stays the safety headroom; CLM simply tries not to
consume it (spec §16, §35, §55, §56).

```text
provider context ceiling:  131K   (what Hermes sees — must be ≥ 64K)
CLM hard limit:            32K    (min(active_target, ceiling*fraction))
CLM usable working target: 28,672 = 32,768 − 4,096 reserve
```

## Multi-agent

`clone_for_agent()` produces an isolated CLM state per agent instance (spec
§25). Subagents cannot mutate the parent's working context; results flow back
through normal Hermes delegation.

## Failure modes

The engine is **fail-open**: any exception inside a hook is swallowed so the
agent never crashes on context management (spec §18). If the model over- or
under-compresses, the deterministic emergency fallback preserves the session.
