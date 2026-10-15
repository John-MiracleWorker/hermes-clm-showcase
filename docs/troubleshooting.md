# Troubleshooting

## Engine not loading

- Confirm the plugin dir exists: `~/.hermes/plugins/hermes-clm/__init__.py`.
- Confirm `context.engine: hermes-clm` in `~/.hermes/config.yaml`.
- The engine `name` must be `clm` (Hermes matches on this).
- The `hermes_clm` package must be importable — `pip install -e .` into the
  Hermes venv.
- The model/provider context must be **≥ 64K**; a smaller advertised window
  is a Hermes startup error, not a CLM error.

## "Nothing was compactable" / `clm_compact` returns `ok: false`

Usually **correct behavior**: with `protect_latest_user_turns=2` +
`protect_latest_assistant_turns=1`, a short conversation has every non-system
message protected, so there is nothing eligible. Grow the conversation or
lower the protection counts to see compaction.

## Restoration rejected (`budget_exceeded`)

The response includes `tokens_required` and `headroom_tokens`. Free space
first (compact/archive stale material) then retry. This is spec §13 working
as intended.

## Active context looks wrong / missing history

Use the recovery path:

```bash
/clm reset-working-set
```

This reconstructs the active working set from canonical history — the
ultimate recovery if CLM state becomes corrupted. Canonical history is never
affected.

## State not persisting / lost on restart

- Confirm `persistence: true` (default).
- State lives under `~/.hermes/clm/<conversation_id>/state.json`.
- If the conversation ID changed (new session), a fresh state root is used by
  design — this is multi-session isolation, not data loss.

## Token counts look off

Token counting falls back in order (spec §34): exact tiktoken → provider
estimator (calibrated from real usage via `update_from_response`) → generic
tokenizer → char-based `chars/3.5` (rounded up, erring high). Install the
`[tiktoken]` extra for exact counts. The counter is calibrated against real
provider prompt tokens, with the system-slot estimate subtracted so tool-schema
overhead doesn't skew the learned ratio.

## Engine appears to do nothing

The engine is **fail-open**: exceptions inside hooks are swallowed so the agent
never crashes. Check the event log (`/clm events` or `events.jsonl`) and the
Hermes debug log for the underlying error. A silent no-op almost always means
an internal exception was caught and logged.

## Tests polluted the real state dir

All tests must run against a throwaway state root (the `conftest.py` fixtures
use `tmp_path`). If you see data under `~/.hermes/clm/`, it came from a manual
run — `rm -rf ~/.hermes/clm/<conversation_id>` to clear it.
