# Benchmark scenarios

Each scenario is a JSON file describing a long-horizon task seed. The runner
expands it into a synthetic transcript (with periodic large "tool output"
turns) and measures token accounting under baseline and CLM handling.

- `long_debug_session.json` — repeated tool calls, large outputs, dead ends,
  eventual diagnosis.

See `docs/benchmarking.md` for the full metric set and the three-config
comparison (A/B/C).
