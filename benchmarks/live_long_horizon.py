#!/usr/bin/env python3
"""Live long-horizon context-engine A/B: hermes-lcm vs hermes-clm + Context Map.

Runs REAL Hermes CLI sessions (one turn per process) against the shared local
Qwen endpoint, chunked and resumable, and records per-turn:

  * wall time and exit status of the `hermes -p <profile> chat --oneshot` call
  * provider usage from the profile's agent.log ("API call #" line)
  * server prefill/decode timing from the Qwen box log window (new lines only)
  * exact-match scoring of delayed-recall probes in the deterministic workload
  * engine-side counters (CLM events.jsonl tail; LCM log markers) per turn

Arms are two Hermes profiles with otherwise-identical configuration:
    lcm   -- context.engine=lcm            (as-deployed default equivalence)
    clm   -- context.engine=hermes-clm + Context Map enabled (production soak config)

This fills the "Phase 2: real long-horizon task run against a local provider"
slot in docs/benchmarking.md, with the LCM comparator arm Trent asked for.

Usage:
    python -m benchmarks.live_long_horizon gen    --seed 14 --turns 240 --out ~/.hermes/clm-bench/lh
    python -m benchmarks.live_long_horizon run    --arm lcm --state ~/.hermes/clm-bench/lh --count 8
    python -m benchmarks.live_long_horizon report --state ~/.hermes/clm-bench/lh

Notes / limits (report these verbatim, not as caveats to bury):
  - A shared single-slot Qwen lane: the runner gates on /slots idleness but a
    foreign consumer can still interleave; box metrics are attributed by the
    new log lines in the turn window and flagged if unmatched.
  - Temperature follows the profile's configured sampling (identical across
    arms); the manifest's temperature-0 pair is a different (paired-fork) rig.
  - No server/model/quantization/launcher changes are made by this harness.

Infra knobs (env; defaults assume one local box): CLM_BENCH_BASE_URL,
CLM_BENCH_DAEMON_URL, CLM_BENCH_BOOT_TOKEN, CLM_BENCH_SSH_HOST,
CLM_BENCH_BOX_LOG, CLM_BENCH_SSL_CERT_FILE.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

BASE = os.environ.get("CLM_BENCH_BASE_URL", "http://127.0.0.1:8081")
DAEMON_URL = os.environ.get("CLM_BENCH_DAEMON_URL", "http://127.0.0.1:8777")
BOX_LOG = os.environ.get("CLM_BENCH_BOX_LOG", r"C:\Users\<user>\prism-llama\qwen-q4-hip-nopeer.log")
SSH_HOST = os.environ.get("CLM_BENCH_SSH_HOST", "box")
PROFILES = {"lcm": "bench-lcm", "clm": "bench-clm"}
SESSION_NAME = "lh-bench"
API_RE = re.compile(r"API call #(\d+): .*?in=(\d+) out=(\d+) total=(\d+) latency=([0-9.]+)s(?: cache=(\d+)/(\d+))?")
SESSION_RE = re.compile(r"^(?:session_id|Session):\s*(\S+)", re.M | re.I)
LOG_SESSION_RE = re.compile(r"(?:INFO \[([0-9]{8}_[0-9]{6}_[0-9a-f]+)\]|\bsession=([0-9]{8}_[0-9]{6}_[0-9a-f]+))")


# --------------------------------------------------------------------------- #
# workload generation (deterministic; identical for both arms)
# --------------------------------------------------------------------------- #

def generate_workload(seed: int, turns: int, out_dir: Path) -> Path:
    import random
    rng = random.Random(seed)
    hexd = lambda n: "".join(rng.choice("0123456789abcdef") for _ in range(n))
    tags = {n: f"TAG-{hexd(12)}" for n in (6, 7)}  # generated once, in fixed order
    rows = []
    values = {}
    for i in range(1, turns + 1):
        value = f"VAL-{hexd(10)}"
        values[i] = value
        parts = [f"[NOTE {i:03d}] field_{i:03d} = {value}"]
        parts.append("filler " + " ".join(hexd(4) for _ in range(40)))
        if i % 10 == 0:  # raw log dump: larger growth turns
            parts.append("raw dump: " + " ".join(hexd(8) for _ in range(400)))
        if i % 26 == 0:  # mega dump: pushes the long-horizon growth budget
            parts.append("mega dump: " + " ".join(hexd(8) for _ in range(900)))
        task = ("append one line 'NOTE %03d' to bench-notes.md in your working directory "
                "(create it if missing), then reply 'noted'." % i if i % 9 == 0
                else "acknowledge with one short sentence.")
        probe = None
        if i >= 40 and i % 7 == 3:
            k = i - 34
            probe = {"kind": "delayed_recall", "field": f"field_{k:03d}",
                     "expected": [values[k]], "delay": 34}
            task = "%s Reply with only the value." % f"Recall drill: what is field_{k:03d} = ?"
        if i in (52, 88):  # opaque-tag probes: tag only ever stated once, at note 006/007
            tag_note = 6 if i == 52 else 7
            probe = {"kind": "opaque_tag", "field": f"opaque_tag_{tag_note:03d}",
                     "expected": [tags[tag_note]], "delay": i - tag_note}
            task = ("%s Reply with only the tag." %
                    f"Recall drill: what opaque tag was noted at note {tag_note:03d}?")
        if i in (6, 7):
            # plant the opaque tags exactly once (values reused for determinism)
            parts.append(f"opaque_tag_{i:03d} = {tags[i]}")
        rows.append({"turn": i, "text": " ".join(parts) + " " + task,
                     "probe": probe, "task": task})
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "workload.jsonl"
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    meta = {"seed": seed, "turns": turns, "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "note": "deterministic; identical text for both arms; tags are hex stand-ins"}
    (out_dir / "workload.meta.json").write_text(json.dumps(meta, indent=1))
    return path


# --------------------------------------------------------------------------- #
# per-turn engine / box plumbing
# --------------------------------------------------------------------------- #

def http_json(path: str, timeout: int = 10):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return json.load(r)


def _boot_qwen_if_down() -> None:
    token_path = Path(os.environ.get("CLM_BENCH_BOOT_TOKEN", str(Path.home() / ".hermes" / "qwen-boot.token")))
    try:
        token = token_path.read_text(encoding="utf-8").strip() if token_path.exists() else ""
        req = urllib.request.Request(DAEMON_URL + "/start", method="POST", data=b"")
        if token:
            req.add_header("X-Auth-Token", token)
        with urllib.request.urlopen(req, timeout=90) as r:
            print(f"[lane_idle] booted qwen daemon (/start -> {r.status})", flush=True)
    except Exception as exc:
        print(f"[lane_idle] qwen daemon /start attempt: {exc}", flush=True)


def lane_idle(wait_s: int) -> bool:
    deadline = time.monotonic() + wait_s
    boot_attempted = False
    while True:
        try:
            slots = http_json("/slots")
            busy = any(isinstance(s, dict) and s.get("is_processing") for s in slots)
            if not busy:
                return True
        except Exception:
            if not boot_attempted:
                boot_attempted = True
                _boot_qwen_if_down()
        if time.monotonic() >= deadline:
            return False
        time.sleep(5)


def pull_box_log() -> str:
    proc = subprocess.run(
        ["ssh", SSH_HOST, "type", BOX_LOG],
        capture_output=True, text=True, timeout=60)
    return proc.stdout


TIMING_RE = re.compile(
    r"task\s+(\d+)\s*\|\s*prompt eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*(\d+)\s*tokens")
EVAL_TASK_RE = re.compile(
    r"task\s+(\d+)\s*\|\s*eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*(\d+)\s*(?:tokens|runs)")


def parse_box_window(text: str, expect_out: int | None, wall_s: float | None = None,
                     watermark: int = 0):
    """Attribute the box's per-request timing to our turn.

    Sequential discipline: only task ids ABOVE `watermark` are candidates
    (stale blocks can share an `out` count with ours). Blocks whose prompt-eval
    exceeds the turn's wall time are impossible for our request and are
    dropped. Among the rest, the closest decode `runs` match to the provider's
    `out` wins (oldest on ties — ours precedes any auxiliary call); fall back
    to the oldest candidate. `n_eval` is the *evaluated* (post-cache) token
    count — the prefill work actually done; wire size comes from the provider.
    """
    prompts = {int(t): (float(ms), int(n)) for t, ms, n in TIMING_RE.findall(text)}
    evals = {int(t): (float(ms), int(n)) for t, ms, n in EVAL_TASK_RE.findall(text)}
    tasks = sorted(set(prompts) | set(evals))
    if not tasks:
        return {"task": None}
    cand = [t for t in tasks if t > watermark] if watermark > 0 else []
    if not cand:
        cand = tasks[-12:]  # no/unknown watermark — best effort recent window
    san = [t for t in cand
           if not (wall_s and t in prompts and prompts[t][0] / 1000.0 > wall_s + 1.0)]
    pool = san or cand
    pick = None
    if expect_out is not None:
        ev = [t for t in pool if t in evals]
        if ev:
            d = min(abs(evals[t][1] - expect_out) for t in ev)
            pick = min(t for t in ev if abs(evals[t][1] - expect_out) == d)
    if pick is None:
        pick = min(pool)
    prompt_ms, n_eval = prompts.get(pick, (None, None))
    eval_ms, n_gen = evals.get(pick, (None, None))
    ok = bool(expect_out is None
              or (pick in evals and abs(evals[pick][1] - expect_out) <= max(3, expect_out // 10)))
    sane = bool(wall_s is None or prompt_ms is None or prompt_ms / 1000.0 <= wall_s + 1.0)
    return {"task": pick, "prompt_eval_ms": prompt_ms, "n_eval": n_eval,
            "eval_ms": eval_ms, "n_gen": n_gen, "attribution_ok": ok and sane,
            "n_candidates": len(cand)}


def tail_agent_log(profile: str, n: int = 60) -> str:
    path = Path.home() / ".hermes" / "profiles" / profile / "logs" / "agent.log"
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def clm_event_count(profile: str, session_id: str | None) -> int | None:
    if not session_id:
        return None
    path = Path.home() / ".hermes" / "profiles" / profile / "clm" / session_id / "events.jsonl"
    try:
        return sum(1 for _ in open(path))
    except OSError:
        return None


def lcm_marker_count(profile: str) -> int | None:
    path = Path.home() / ".hermes" / "profiles" / profile / "logs" / "agent.log"
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return None
    return len(re.findall(r"LCM[ -]?(compaction|preflight|rewrite)", text, re.I))


# --------------------------------------------------------------------------- #
# run / report
# --------------------------------------------------------------------------- #

def run_arm(args) -> int:
    import fcntl
    state = Path(args.state).expanduser()
    state.mkdir(parents=True, exist_ok=True)
    lock_fh = open(state / f".runner-{args.arm}.lock", "w")
    try:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(f"[run_arm] another runner already holds .runner-{args.arm}.lock; exiting.", flush=True)
        return 1
    lock_fh.write(str(os.getpid()) + "\n")
    lock_fh.flush()
    profile = PROFILES[args.arm]
    workload = [json.loads(line) for line in (state / "workload.jsonl").read_text().splitlines()]
    results_path = state / f"results-{args.arm}.jsonl"
    done = set()
    if results_path.exists() and args.resume:
        for line in results_path.read_text().splitlines():
            try:
                done.add(json.loads(line)["turn"])
            except Exception:
                pass
    session_id = None
    if results_path.exists():
        for line in reversed(results_path.read_text().splitlines()):
            try:
                session_id = json.loads(line).get("session_id") or session_id
            except Exception:
                pass
    prev_box = state / f"boxlog-{args.arm}.txt"
    wm_path = state / f"last_task-{args.arm}.json"
    watermark = 0
    if wm_path.exists():
        try:
            watermark = int(json.loads(wm_path.read_text()).get("last_task", 0) or 0)
        except Exception:
            watermark = 0
    hermes_bin = shutil.which("hermes") or str(Path.home() / ".local" / "bin" / "hermes")
    executed = 0
    run_start = time.monotonic()
    with open(results_path, "a") as fh:
        for row in workload:
            if executed >= args.count:
                break
            if row["turn"] in done:
                continue
            if not lane_idle(args.wait_lane):
                print(f"[turn {row['turn']}] lane busy beyond wait budget; stopping.")
                break
            cmd = [hermes_bin, "-p", profile, "chat", "-c", SESSION_NAME,
                   "--create-if-missing", "--oneshot", "-Q", "-q", row["text"]]
            t0 = time.monotonic()
            try:
                env = os.environ.copy()
                env["HOME"] = os.path.expanduser("~")
                env["HERMES_ACCEPT_HOOKS"] = "1"
                ca = os.environ.get("CLM_BENCH_SSL_CERT_FILE")
                if ca:
                    env["SSL_CERT_FILE"] = ca
                proc = subprocess.run(cmd, capture_output=True, text=True,
                                      timeout=args.turn_timeout, cwd=str(state), env=env)
                rc, out, err = proc.returncode, proc.stdout, proc.stderr
            except subprocess.TimeoutExpired as exc:
                out = exc.stdout or ""
                err = exc.stderr or ""
                if isinstance(out, bytes):
                    out = out.decode(errors="replace")
                if isinstance(err, bytes):
                    err = err.decode(errors="replace")
                rc, err = -9, err + "\nTIMEOUT"
            wall = time.monotonic() - t0
            reply = out
            m = SESSION_RE.search(out)
            if m:
                session_id = m.group(1)
                reply = out[:m.start()]
            reply = reply.strip()
            api = None
            am = None
            for line in reversed(tail_agent_log(profile, 120).splitlines()):
                if not session_id:
                    sm = LOG_SESSION_RE.search(line)
                    if sm:
                        session_id = sm.group(1) or sm.group(2)
                am = API_RE.search(line)
                if am:
                    if not session_id:
                        sm = LOG_SESSION_RE.search(line)
                        if sm:
                            session_id = sm.group(1) or sm.group(2)
                    api = {"call": int(am.group(1)), "in": int(am.group(2)),
                           "out": int(am.group(3)), "total": int(am.group(4)),
                           "latency_s": float(am.group(5))}
                    if am.group(6) is not None:
                        api["cache_reused"] = int(am.group(6))
                        api["cache_total"] = int(am.group(7))
                    break
            time.sleep(2)  # let the server finish writing the log
            box_text = pull_box_log()
            prev_box.write_text(box_text, errors="replace")  # audit mirror
            box = parse_box_window(box_text, api["out"] if api else None,
                                   wall_s=wall, watermark=watermark)
            if box.get("task"):
                watermark = box["task"]
                try:
                    wm_path.write_text(json.dumps({"last_task": watermark}))
                except Exception:
                    pass
            score = None
            if row.get("probe"):
                exp = row["probe"]["expected"]
                score = {"kind": row["probe"]["kind"], "expected": exp,
                         "matched": any(e.casefold() in reply.casefold() for e in exp)}
            record = {"turn": row["turn"], "arm": args.arm, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                      "wall_s": round(wall, 1), "rc": rc, "reply": reply[:2000],
                      "reply_chars": len(reply), "session_id": session_id,
                      "api": api, "box": box, "score": score,
                      "clm_events": clm_event_count(profile, session_id) if args.arm == "clm" else None,
                      "lcm_markers": lcm_marker_count(profile) if args.arm == "lcm" else None,
                      "lane": "shared"}
            fh.write(json.dumps(record) + "\n")
            fh.flush()
            executed += 1
            flag = ""
            if score:
                flag = " PROBE " + ("PASS" if score["matched"] else "FAIL")
            print(f"[turn {row['turn']:03d}] wall={wall:.1f}s rc={rc} "
                  f"in={(api or {}).get('in', '?')} pev={box.get('prompt_eval_ms')}ms "
                  f"task={box.get('task')} sid={session_id} ev={record['clm_events']}{flag}", flush=True)
            if args.wall_budget and (time.monotonic() - run_start) > args.wall_budget:
                print("wall budget exhausted; stopping")
                break
    return 0


def report(state_dir: Path) -> str:
    lines = ["# live long-horizon lcm-vs-clm report", ""]
    for arm in PROFILES:
        path = state_dir / f"results-{arm}.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        if not rows:
            continue
        ins = [r["api"]["in"] for r in rows if r.get("api")]
        walls = [r["wall_s"] for r in rows]
        pev = [r["box"]["prompt_eval_ms"] for r in rows if (r.get("box") or {}).get("prompt_eval_ms")]
        nev = [r["box"]["n_eval"] for r in rows if (r.get("box") or {}).get("n_eval")]
        evs = [(r["box"]["eval_ms"], r["box"]["n_gen"]) for r in rows
               if (r.get("box") or {}).get("eval_ms") and (r.get("box") or {}).get("n_gen")]
        tps = [n / (ms / 1000.0) for ms, n in evs if ms]
        probes = [r for r in rows if r.get("score")]
        passed = [r for r in probes if r["score"]["matched"]]
        reuse = [(i - n) / i for i, n in zip(ins, nev) if i and n and n <= i]
        lines += [
            f"## arm `{arm}` (profile {PROFILES[arm]})",
            f"- turns: {len(rows)} | errors: {sum(1 for r in rows if r['rc'] != 0)}",
            f"- wire in-tokens: first {ins[0] if ins else '?'} / last {ins[-1] if ins else '?'} "
            f"/ max {max(ins) if ins else '?'}",
            f"- wall s: mean {sum(walls)/len(walls):.1f} / max {max(walls):.1f}",
            f"- prompt-eval ms: n={len(pev)} mean {sum(pev)/len(pev):.0f} / sum {sum(pev):.0f}"
            if pev else "- prompt-eval ms: n/a",
            f"- decode tok/s: mean {sum(tps)/len(tps):.1f}" if tps else "- decode tps: n/a",
            f"- cache reuse ratio: mean {sum(reuse)/len(reuse):.3f}" if reuse else "- reuse: n/a",
            f"- probes: {len(passed)}/{len(probes)} pass",
            "",
        ]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--seed", type=int, default=14)
    g.add_argument("--turns", type=int, default=240)
    g.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arm", choices=PROFILES.keys(), required=True)
    r.add_argument("--state", required=True)
    r.add_argument("--count", type=int, default=8)
    r.add_argument("--resume", action="store_true", default=True)
    r.add_argument("--turn-timeout", type=int, default=1200)
    r.add_argument("--wait-lane", type=int, default=300)
    r.add_argument("--wall-budget", type=int, default=0)
    p = sub.add_parser("report")
    p.add_argument("--state", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "gen":
        path = generate_workload(args.seed, args.turns, Path(args.out).expanduser())
        print("workload:", path)
        return 0
    if args.cmd == "run":
        return run_arm(args)
    if args.cmd == "report":
        print(report(Path(args.state).expanduser()))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
