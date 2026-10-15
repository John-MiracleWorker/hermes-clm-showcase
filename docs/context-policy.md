# Context Policy

The policy layer (`policies.py`) is the single place that decides whether a
model-proposed context operation is structurally valid. The model proposes;
the host validates (spec §4.3, §41).

## Protection (spec §8)

**Always protected (never compactable/archivable):**
- system prompt,
- original root task,
- global safety instructions,
- Hermes tool contract,
- CLM tool contract.

**Configurable protection (recent-turn retention):**
- `protect_latest_user_turns` (default 2),
- `protect_latest_assistant_turns` (default 1).

Any CLM operation referencing a protected item is rejected with:

```json
{ "ok": false, "error": "protected_context", "message_ids": ["..."] }
```

## Validation (spec §32.6)

Operations are rejected when they:

- reference unknown message IDs (`unknown_message_ids`),
- overlap protected content (`protected_context`),
- are empty after validation (`empty_operation`),
- exceed the hard budget (`budget_exceeded`, with `tokens_required` +
  `headroom_tokens` reported),
- remove the latest required user instruction.

## Atomicity (spec §33)

Every mutating operation takes a store **snapshot** before applying. If any
step fails, the store is rolled back to the snapshot — state is never left
partially modified. `clm_compact` validates the *whole* ID set before archiving
anything, so a single bad ID aborts the entire operation.

## Nudging (spec §17)

Escalating notices as utilization rises (default 0.25 / 0.50 / 0.75 / 0.90):
- **25–50%** — informational, "you have headroom".
- **75%** — encourage proactive cleanup.
- **90%** — strongly recommend immediate context management.

The model is never forced to compact merely for crossing a threshold (spec
§15): "Your goal is task success, not maximum compression."

## Budget-aware restore (spec §13)

`clm_restore` is budget-aware. If restoring would exceed the hard limit, the
request is rejected and the engine reports the exact `tokens_required` and
`headroom_tokens` so the model can free space first (compact/archive) and
retry.

## Emergency fallback (spec §19)

When the model fails to manage context and overflow retries are exhausted, a
deterministic host policy runs, in order:

1. remove already-summarized raw spans,
2. remove old low-value tool outputs,
3. preserve latest user/assistant work,
4. preserve unresolved decisions,
5. preserve all protected messages,
6. produce an emergency host summary if still necessary.

Emergency compression is clearly labeled in telemetry
(`emergency_compaction`), and the canonical conversation is always preserved.

## Prompt-injection resistance (spec §32.5)

CLM policy lives in the **protected system layer**. Tool outputs and retrieved
documents can be archived by the model, but they can never redefine protected
rules, retention, tool permissions, or system instructions.
