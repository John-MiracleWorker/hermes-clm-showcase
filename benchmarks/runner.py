"""Benchmark runner (spec §43/§45) — offline, stdlib-only.

Drives the CLM engine (and a no-CLM baseline) against a long synthetic
transcript and captures the token accounting the core hypothesis needs:
peak / mean / total prompt tokens, context edits, and emergency compactions.

A real long-horizon task run against a local provider is the Phase 2 deliverable;
this harness proves the measurement path end-to-end without a model call.

Usage:
    python -m benchmarks.runner --scenario scenarios/long_debug_session.json \
        [--clm-target 32768]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.environ.get("HERMES_AGENT_PATH", os.path.expanduser("~/.hermes/hermes-agent")))

from hermes_clm import CLMConfig, CLMEngine  # noqa: E402


def _load_scenario(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _build_turns(spec: dict):
    """Expand a compact scenario spec into a full message list."""
    n = spec.get("turns", 60)
    pad = spec.get("char_pad", 300)
    seed = spec.get("seed", "debug")
    msgs = [{"role": "system", "content": "You are a debugging assistant."}]
    for i in range(n):
        # Mix in occasional large 'tool output' turns to simulate bloat.
        is_tool = (i % 5 == 0)
        body = (f"{seed} step {i} " + ("T" if is_tool else "x") * (pad * (4 if is_tool else 1)))
        msgs.append({"role": "user", "content": f"user {i} " + body[:200]})
        msgs.append({"role": "assistant", "content": f"asst {i} " + body})
    return msgs


def _baseline_metrics(msgs):
    """No-CLM: every turn re-sends the whole accumulated transcript."""
    counter_chars = sum(len(m.get("content", "")) for m in msgs)
    # Standard Hermes carries the full history each call.
    running = []
    total = 0
    peak = 0
    for m in msgs:
        running.append(m)
        c = sum(len(x.get("content", "")) for x in running)
        total += c
        peak = max(peak, c)
    return {
        "mode": "baseline",
        "turns": len(msgs),
        "final_prompt_chars": len(running) and sum(len(m.get("content", "")) for m in running),
        "total_prompt_chars": total,
        "peak_prompt_chars": peak,
        "context_edits": 0,
        "emergency_compactions": 0,
    }


def _clm_metrics(msgs, clm_target):
    """Faithful per-turn simulation of the host contract (spec §7).

    The host calls ``select_context(full_conversation)`` before every provider
    request. We measure what the model actually sees each turn, accumulate
    prompt tokens, and let the model compact when utilization crosses 0.75.
    """
    tmp = tempfile.mkdtemp(prefix="clm-bench-")
    cfg = CLMConfig()
    cfg.state_root = os.path.join(tmp, "clm")
    cfg.active_target_tokens = clm_target
    cfg.emergency_host_fallback = True
    eng = CLMEngine(config=cfg, hermes_home=tmp)
    eng.on_session_start("sess-bench", conversation_id="conv-bench")

    from hermes_clm.tools import dispatch
    import json as _j

    total_selected = 0
    peak = 0
    edits = 0
    n = len(msgs)
    for i in range(1, n + 1):
        prefix = [m.copy() for m in msgs[:i]]
        selected = eng.select_context(prefix) or prefix
        # observe() so peak/mean metrics reflect the working set (spec §31).
        eng.budget.observe(eng.store.working_set_tokens())
        sel_tokens = eng.counter.count_messages(selected)
        total_selected += sel_tokens
        peak = max(peak, sel_tokens)

        # Simulate the model managing context at every 8th turn when it's full.
        if i % 8 == 0:
            working = eng.store.working_set_tokens()
            if eng.budget.utilization(working) >= 0.75:
                targets = [r["uid"] for r in eng.list_active()
                           if r["role"] == "assistant" and not r["protected"]][:6]
                if targets:
                    res = _j.loads(dispatch(eng, "clm_compact", {
                        "message_ids": targets,
                        "summary": f"Condensed debugging steps ~{i}."}))
                    if res.get("ok"):
                        edits += 1

    kinds = {e.get("event") for e in eng.recent_events(500)}
    emergency = 1 if "emergency_compaction" in kinds else 0
    final = eng.counter.count_messages(
        eng.select_context([m.copy() for m in msgs]) or [])
    return {
        "mode": "clm",
        "clm_target_tokens": clm_target,
        "turns": n,
        "final_active_tokens": final,
        "peak_selected_tokens": peak,
        "total_selected_tokens": total_selected,
        "mean_selected_tokens": total_selected // max(1, n),
        "peak_working_set_tokens": eng.budget.metrics_snapshot()["peak_active_context_tokens"],
        "context_edits": edits,
        "emergency_compactions": emergency,
        "summaries": len(eng.store.summaries),
        "archived": len(eng.store.archived_messages()),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--clm-target", type=int, default=32768)
    ap.add_argument("--out", default="benchmarks/reports")
    args = ap.parse_args(argv)

    if args.scenario and os.path.exists(args.scenario):
        spec = _load_scenario(args.scenario)
    else:
        spec = {"turns": 60, "char_pad": 300, "seed": "debug"}

    msgs = _build_turns(spec)
    baseline = _baseline_metrics(msgs)
    clm = _clm_metrics(msgs, args.clm_target)

    report = {"scenario": os.path.basename(args.scenario) if args.scenario else "default",
              "baseline": baseline, "clm": clm}
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"\nreport written: {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
