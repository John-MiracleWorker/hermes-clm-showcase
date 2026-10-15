# Benchmarking

The central hypothesis (spec §46):

> Can Hermes + CLM keep its **active working set near 32K** while running on a
> **64K+ provider window**, and achieve similar task success to normal Hermes
> carrying a much larger 131K context — while processing substantially fewer
> prompt tokens?

## Configurations (spec §43)

| Config | Provider/model context | Context handling |
|--------|------------------------|------------------|
| **A** | 64K | standard Hermes |
| **B** | 64K (or 131K) + 32K CLM target | Hermes + CLM |
| **C** | 131K | standard Hermes |

Config B may also run at a 131K ceiling to isolate self-managed context from
raw window size.

## Tasks (spec §44)

Long-horizon tasks that naturally accumulate context:

- **Repository engineering** — inspect, implement, test, debug, document.
- **Research** — investigate, compare, gather evidence, synthesize.
- **Multi-agent project** — parent delegates research/coding/testing.
- **Long debugging session** — repeated tool calls, dead ends, diagnosis.

Each scenario is a reproducible seed under `benchmarks/scenarios/`.

## Metrics (spec §45)

Primary: `task_success`, `total_prompt_tokens`, `average_context_tokens`,
`peak_context_tokens`, `wall_clock_time`.

Secondary: `ttft`, `tokens_per_second`, `tool_calls`, `context_edits`,
`restores`, `emergency_compactions`, `wrongly_archived_information`,
`task_rework_due_to_missing_context`.

## Running

The runner loads a scenario manifest, drives Hermes (with and without CLM),
captures token/latency telemetry from `events.jsonl` + provider usage, and
writes a comparative report to `benchmarks/reports/`.

```bash
python -m benchmarks.runner --scenario scenarios/repo_engineering.yaml \
    --config B --provider http://127.0.0.1:8081/v1
```

## Pass condition (spec §55)

A long task finishes while **active context remains materially below the raw
accumulated transcript size**, with no host emergency truncation, and task
success comparable to the non-CLM baseline.

> Note: the full runner is a Phase 2 deliverable. The v0.1 telemetry
> (`events.jsonl` + budget metrics) is what makes the comparison reproducible.
