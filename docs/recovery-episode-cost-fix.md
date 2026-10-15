# Recovery-episode cost — design brief & fix record (2026-10-14)

**Context:** implementation brief for the recovery-episode cost fix. The fix landed the same day (frozen episode presentation + deferred rewrites; full validation in the benchmark report). This document is retained as the design + forensics record.
**Repo:** this repository (branch `work/context-map`).
**Status of system:** CLM is stable and functionally correct (probes 17/17 vs Hermes-LCM's 15/17 in the same window). The remaining problem is **cost**: recovery episodes consume ~77% of total run wall time.

---

## 1. TL;DR — the finding you are being asked to fix

A partial 340-turn live benchmark (`run4`) was killed at t151. Measured facts:

- **18 recovery turns armed in 151 turns**, 17 model-executed `compact` ops, 16 settles, 6 `overflow` + 6 `trim` events.
- **10 expensive episodes**; 22 slow turns (>60s) = 6,801s of a **146.9 min total wall (77%)**.
- Per-call server forensics (from the box log) show the cost mechanism: **during an episode, nearly every consecutive model call re-prefills the ENTIRE body (~70–92k tokens) instead of evaluating a delta.**

  Example — turn 130 (wall 863.7s). All server tasks between turn 129's last call and turn 130's last call:
  ```
  task 20268: pe 3.89s / 1501 tok     <- normal delta (t129)
  task 20313: pe 95.42s / 79,194 tok  <- FULL re-prefill  (call 1 of t130)
  task 21203: pe 86.83s / 73,164 tok  <- FULL re-prefill  (call 2)
  task 22111: pe 4.39s / 2,228 tok    <- delta (cache hit!) (call 3)
  task 22154: pe 111.27s / 89,293 tok <- FULL re-prefill  (call 4)
  task 24101: pe 114.64s / 91,847 tok <- FULL re-prefill  (call 5)
  ```
  Turn 150–151 (walls 330s / 243s):
  ```
  task 30914: 3.87s / 1,491     (t149 normal)
  task 30941: 84.77s / 71,680   FULL
  task 31533: 11.64s / 7,056    delta-ish
  task 32147: 82.87s / 70,292   FULL   (t150)
  task 32501: 86.90s / 73,231   FULL   (t151)
  task 33115: 80.47s / 68,696   FULL   (t151)
  ```

- Full prefill runs at ~800–860 tok/s on this box, so **each full body re-prefill costs 80–115s**. Episode cost ≈ `N_calls × full_prefill`. A clean turn costs 0.6–6s of prefill.
- **Cache reuse demonstrably works when consecutive prompts render identically** (the `22111` / `31533` delta hits are proof) — so the fix space is: make episode-scoped renders byte-stable, or eliminate the calls.

**Goal:** episodes should cost ≈ ONE body re-prefill (the unavoidable post-rewrite one), not 4–5. Stretch: full 340 turns ≤ ~90 min wall (LCM does 340 in 86 min).

---

## 2. Where the machinery lives

`hermes_clm/recovery.py` (544 lines) — the whole recovery-episode system:
- `build_recovery_turn_directive()` L40 — directive text; includes candidates list **re-measured "right now" per render** and an `{remaining}` attempts counter that mutates.
- `class RecoveryTurn` L96 — state machine: `arm()` L125, `mark_emitted()` L159, `observe_op()` L163, `settle()` L174.
- `class RecoveryTurnMixin` L244 — host integration: `_maybe_arm_recovery_turn()` L275, `_observe_host_pressure()` L313, `_recovery_goal_met_in_turn()` L345, `_recovery_directive_suffix()` L360, `_recovery_candidates()` L373, `_recovery_tail_echo()` L408, `_observe_recovery_tool()` L431, `_settle_recovery_turn()` L444, `_recovery_fallback_archive()` L487.

`hermes_clm/engine.py` hooks:
- L530–539: turn boundary settles an `offered` recovery turn (`trigger='turn_complete'`).
- L~640: arms from measured pressure (production host path).
- L686–691: **while `offered`, the directive "rides this preparation's notice"** — i.e. the directive text is appended to the notice (prompt-top region), in addition to the tail echo.
- L~700: overflow path trims to fit (spec §18 step 2) and records `trim` telemetry.
- L713: return path = `_attach_recovery_tail_echo(_attach_context_map(selected))`.

Tools (model-facing): `hermes_clm/tools/` — `clm_compact` (replace span with summary), `clm_archive`, `clm_status`, `clm_lookup`, `clm_restore`, `clm_replace_summary`. Note `tools/__init__.py:171` special-cases `clm_compact` / `clm_archive`.

Config (`hermes_clm/config.py`): `active_target_tokens` L47 (default 32,768; profile sets reserve 8,192 → usable 24,576), `emergency_host_fallback` L71 (default True — host-side deterministic fallback, currently only fires when a recovery turn makes no measurable progress), `recovery_turn` L76 (default False; enabled in the bench profile).

Telemetry: `~/.hermes/profiles/bench-clm/clm/<session>/events.jsonl`. Run4 session = `20261014_114859_ebd5a5` (366 events: 18 `recovery_turn` [initiator host], 17 `compact` [initiator model], 16 `recovery_turn_settled` [host], 6 `overflow`, 6 `trim`, 153/150 session_start/end — one CLI process per benchmark turn).

Harness (the thing that drove the run): `benchmarks/live_long_horizon.py`. One `hermes -p bench-clm chat -c lh-bench --oneshot` process per turn; per-turn kill cap `--turn-timeout` default **1200s** → SIGKILL, recorded `rc=-9` (t98 hit exactly this at 1200.0s).

---

## 3. Hypotheses, ranked, each with how to CONFIRM before patching

**H1 — Episode-scoped render instability (most likely).** The directive/notice/tail-echo content is re-rendered per API call while a recovery turn is `offered`, and mutates between calls: candidates list is re-measured "right now" (L373), the `{remaining}` admission counter changes (L40-74), `observe_op()` updates state after each tool op (L163), and settle edits/retires the directive (L444+, engine L530-539). Any mutation above the tail diverges the prefix → llama-server re-prefills the whole body. The tail-echo mechanism (L408) exists precisely to confine mutations to the tail — the evidence says it is not achieving that for most calls.
*Confirm:* in `prepare_request`/selector build path, log per call: a hash of the fully rendered prompt + the byte offset of the first divergence vs the previous call's prompt + the wire length. Correlate divergence offsets with full-prefill events in the box log. If divergences are at the notice/directive position (top) rather than the tail, H1 is confirmed.

**H2 — Mid-turn compact rewrites.** The model executes `clm_compact` during its turn; the rewrite replaces a span somewhere in history, invalidating the prefix from the rewrite position for every subsequent call that turn (and the settle render for the next turn).
*Confirm:* map each server task in an episode window (box log "task N | prompt eval time = …") to its call index and the intervening tool op; compare `n_eval` against the position of the compacted span. If fulls align 1:1 with post-compact calls, confirmed.
*Fix direction:* defer the physical history rewrite to the turn boundary (host-side), so the current turn's remaining calls stay warm and only the next turn pays one re-prefill. (See H5: the host-side path already exists.)

**H3 — Call-count per episode.** 4–5 LLM calls per episode, each paying cost. The directive already lists reclaimable candidates inline to avoid a `clm_status` hop — but the model still round-trips (compact → result → verify → maybe more).
*Confirm:* per-episode call counts from `api.call` fields + box-log task windows (t130 = 5 calls; t150 = 3; t151 = 2).
*Fix direction:* allow/encourage issuing **all** compacts for an episode in one assistant message (batched tool calls), or have the host execute the candidate list in one shot.

**H4 — Trigger depth.** Episodes fire at 70–92k wire; rewrite+reprefill cost scales with depth. Peaks drifted upward across the run (66k → 70k → 91k → 92k).
*Confirm:* scatter of trigger wire vs episode wall. Lower priority than H1/H2 — do not tune thresholds before the per-call stability fix, or you will mask the real signal.

**H5 — Settle/fallback duplication.** `emergency_host_fallback` (default True) runs "exactly once" when the model's turn fails to reduce; check whether it adds a second rewrite + extra full prefill on top of a partially-effective model turn (events carry `archived`, `restore_point`, `store_tokens_*` fields).
*Confirm:* compare `store_tokens_at_arm/settle/after` + task counts for episodes with/without fallback in the event log.

---

## 4. Evidence files (public copies live under `data/` in this repo)

| File | Contents |
|---|---|
| `data/results-clm.jsonl` | Authoritative per-turn records (wall, rc, wire, box timings, api.call/in, reply, clm_events); turns 1–151 are the run4 slice referenced here |
| `data/boxlog-clm.txt` | Box llama-server log capture (per-task prompt/eval timings, draft acceptance) |
| `data/runner-clm.log` | Per-turn runner lines incl. PROBE PASS markers |
| `data/results-lcm.jsonl` | LCM comparison arm (same workload, complete 340) |
| `data/workload.jsonl` | Deterministic 340-turn workload w/ probes (`probe` field: kind/field/expected/delay) |
| Session events | engine-side `events.jsonl` (recovery_turn / compact / settle counters; private state, not included) |

Box-log attribution rules (watermark per turn = task id) live in the harness `parse_box_window` docstring.

Run4 summary table (turns 1–151):

| Metric | CLM run4 | LCM same window |
|---|---|---|
| Wall | 146.9 min (mean 58.4s) | 37.3 min (mean 14.8s) |
| Fast turns ≤30s | 122 @ 16.5s | 146 @ 12.6s |
| Prefill total | 2,124s | 522s |
| Decode | 27.9 t/s | 29.9 t/s |
| Wire max | 91,847 (t130) | 112,365 (t100) |
| Probes ≤151 | 17/17 | 15/17 |
| Errors | 1 (t98, cap-killed) | 0 |

---

## 5. Ground rules — do NOT break these

1. **Turn-path cache parity is the crown jewel.** Normal turns render byte-stable; delta prefills of 0.6–6s and 10–16s walls depend on it. Your fix must not touch normal-path render stability. (Previous fix: `8947780` detached-fork cache parity — do not regress.)
2. **Probe retention.** 17/17 depends on model-authored summaries preserving exact `key=value` facts (the directive says so; keep). Any move to pure deterministic host summarization must be checked against probe drills in the workload.
3. **Bounded, evidence-first, fail-closed.** Keep the existing test suites green; add tests for whatever you change:
   `.venv/bin/python -m pytest tests/test_recovery_turn_v1.py tests/test_retrofit_ghost_retirement.py tests/test_context_map_lookup_restore_v1.py tests/test_compaction.py tests/test_message_envelopes.py -q`
   (The full suite has ~679 pre-existing unrelated failures; baseline is recorded in `tests/` history — only ensure zero NEW failures.)
4. **Box constraints.** 32 GB RAM; stable-served envelope ≤ ~112k wire (a 120k walk crashed the box twice historically); single lane at `<box>:8081`; all client traffic was serialized through a local FIFO broker (`127.0.0.1:8181`, one request at a time); the lane was shared with other agents — do not disturb broker/box config.
5. **Archive discipline.** Before any rerun: copy (not move) `results-clm.jsonl`, `boxlog-clm.txt`, `last_task-clm.json` to `*.runN-partialM.bak`. Remove `.runner-clm.lock` only when no runner lives.
6. **Harness per-turn cap** is 1200s (`--turn-timeout`); a fixed episode must comfortably fit — target ≪ that.

---

## 6. Verification protocol (how your fix will be judged)

1. **Unit:** new tests covering the byte-stability/freeze behavior you implement (e.g. "directive renders identical across calls within an episode", "no non-append mutation above tail while offered").
2. **Live A/B (preferred):** resume the existing run —
   `.venv/bin/python -u -m benchmarks.live_long_horizon run --arm clm --state <state-dir> --count 340`
   It continues from t152 with session/CLM state intact (fresh lh-bench session must NOT be created; the run command resumes by logged count).
   Compare against the run4 window using box-log truth: per-episode full-prefill count and episode wall.
   **Pass bar:** ≤1 full body re-prefill per episode (vs 4–5 now); episode wall ≤ ~120s (vs 215–1,200s); no regression in probe passes or normal-turn walls.
3. **Report format:** per-call box-log evidence (prompt eval ms/tokens, eval ms, draft acceptance), before/after tables, exact commit hash pushed to `work/context-map`.

---

## 7. Environment cheatsheet

```bash
# health / state
curl -s http://<box>:8081/health
curl -s http://<box>:8081/slots
curl -s http://127.0.0.1:8181/status          # broker
# box ssh
ssh <box>   # box log: C:\Users\<user>\prism-llama\qwen-q4-hip-nopeer.log
# bench
cd ~/hermes-clm
.venv/bin/python -m benchmarks.live_long_horizon report --state ~/.hermes/clm-bench/lh
# episode forensics (per-call windows in box log)
# per-call episode windows parsed from the box log via parse_box_window
```

Constraints recap: macOS arm64 host; Python 3.12 venv `.venv`; commit + push to `work/context-map` when green.

---

## 8. Ready-to-paste brief

> CLM recovery-episode cost fix — this repository, branch `work/context-map`. Read `docs/recovery-episode-cost-fix.md` (full context, evidence, hypotheses, guardrails, verification protocol).
>
> Problem: in a 151-turn live run, recovery episodes consumed 77% of wall time because nearly every consecutive model call inside an episode re-prefills the entire 70–92k-token body (~80–115s each, 4–5× per episode) instead of evaluating a delta. Cache reuse works whenever consecutive renders are byte-identical (proof: delta hits mid-episode), so the fix is render stability of episode-scoped content (directive/notice/tail-echo/candidates/attempts counter) and/or eliminating calls (host-side or batched compacts), possibly deferring physical history rewrites to turn boundaries.
>
> Mandate: instrument first (log per-call prompt hash + first-divergence offset vs previous call), prove the divergence source, then fix with tests. Do not regress normal-turn cache parity (delta prefills ~0.2–2.3k tok) or probe retention (17/17). Verify by resuming the benchmark (`run --arm clm --count 340` continues at t152) and comparing per-episode full-prefill counts and walls via box-log truth. Pass bar: ≤1 full re-prefill per episode, episode wall ≤ ~120s. Deliverables: green targeted tests, before/after box-log evidence, commit pushed to `work/context-map`. Full evidence files and exact commands are in the handoff doc.
