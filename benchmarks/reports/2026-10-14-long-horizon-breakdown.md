# Long-Horizon Context Benchmark — hermes-lcm vs hermes-clm + Context Map

**Live run:** 2026-10-14 (EDT), single lane, local Qwen3.8-27B-Q4_K_M on the dual-RX-9070 box
**Scope:** 340-turn agent workload, two arms (`bench-lcm` = stock Hermes + hermes-lcm; `bench-clm` = hermes-clm + Context Map)
**Status:** Both arms complete (340/340): LCM 2026-10-14 ~05:05; CLM 2026-10-14 21:43 EDT (recovery-episode cost fix landed mid-run — updates below).
**Raw data:** `data/` in this repo (`results-lcm.jsonl`, `results-clm.jsonl`, `workload.jsonl`, `boxlog-*.txt`)
**Report generated:** 2026-10-14 ~05:10 EDT; **CLM column finalized 2026-10-15 ~09:00 EDT** after the full 340-turn arm completed.

> **ERRATA 2026-10-14 ~11:00:** Post-run forensics (see `docs/recovery-episode-cost-fix.md`) found the pre-fix CLM arm ran with 5 compounding bugs — ghost-freeze from an envelope mismatch and a token-calibration inflation that injected ~29.3k extra tokens after the first API call of every tool loop (exactly the ~30k reuse-boundary signature measured in §5). **Treat the CLM column as measuring a broken engine, not CLM-as-designed.** A clean 340-turn rerun completed 2026-10-14 21:43 (session `20261014_114859_ebd5a5`).
>
> **UPDATE 2026-10-15 ~09:00:** §5/§7 now carry the final complete CLM column. The recurring full-body re-prefill regime described in the original §5 belongs to the pre-fix era; the 108-turn crash-stopped dataset is preserved in git history.

---

## 1. TL;DR

- **Both engines accumulate context essentially identically** until the ceiling: wire tokens tracked within ~1% at every 20-turn checkpoint (e.g. t100: LCM 112,365 vs CLM 111,365). The engines differ in what happens *around* the ceiling.
- **LCM (complete, 340 turns, 0 errors):** stays delta-only between rare rewrites. 4 compactions total (~111k → ~50–59k each), each costing 25–74 s of engine passes + a 2–3 turn re-prefill cliff (60–134 s walls). Recall: **39/44 probes — all 5 misses had a compaction between plant and ask (5/13 crossing probes lost, 0/31 non-crossing).**
- **CLM (complete, 340 turns, 2 errors):** with the recovery-episode cost fix (landed mid-run), each pressure episode costs **≤1 bounded re-prefill** (5–30 k tokens) plus tail deltas — the pre-fix regime of recurring full-body re-prefills (40–124 s each, every ~10 turns) is gone. Recall: **41/44** (2 genuine misses; 1 timeout casualty).
- **Net cost (full run):** per turn, CLM spent **2.76× the prefill time and 3.55× the wall time** of LCM (11,013 ms vs 3,986 ms prefill/turn; 54.3 s vs 15.3 s wall/turn — CLM's mean includes the pre-fix era + two cap events; post-fix stretch t163–340: 50.9 s). Decode: CLM 32.8 vs LCM 30.7 t/s.
- **Risk/wire:** with recovery active (final config), CLM's wire cycled in a stable band **~50 → 92 k (max 91,847)** and never re-entered the pre-fix 120 k crash zone (LCM max 112,608). Two recovery-turn **runaway generations** occurred (~2 per 250 turns); both were absorbed by the harness 20-min cap + llama-server cancel-on-disconnect.

## 2. What was tested and why

The CLM design hypothesis (spec §46, `docs/benchmarking.md`): run long-horizon work with a **small active working set** (target ~32–49 k) on a large window, processing fewer prompt tokens than a standard engine carrying the full context. This run evaluates that claim **as-deployed** on the production rig:

| Arm | Profile | Engine | Configuration |
|---|---|---|---|
| LCM | `bench-lcm` | hermes-lcm (lossless context mgmt) | as production default; compaction threshold 0.85 (fires ≈ 111.4 k) |
| CLM | `bench-clm` | hermes-clm + Context Map | `recovery_turn: true`, `active_target_tokens: 49152`, `context_map_enabled: true` (production-soak config) |

Both profiles: same model (`Qwen3.8-27B-Q4_K_M.gguf`), same provider, one conversation each ("lh-bench"), driven by the same harness through the same workload, strictly serial on the single lane (np1 / 131 k / MTP2 / `--cache-ram 10240`).

## 3. Methodology

**Workload** (`~/.hermes/clm-bench/lh/workload.jsonl`, seed 14, deterministic, extensible):
- 340 turns, ~240 tokens/turn average growth via notes + filler (~327 k chars ≈ 82 k tokens of content).
- Raw dumps every 10th turn; mega-dumps every 26th.
- **44 probes**, two kinds: *delayed-recall* values (planted, asked exactly **34 turns later**) and *opaque tags* (planted t6; checked t52/88). Score = whether the reply reproduces the expected value.
- Audit note: turns ≤ 6 (LCM) / ≤ 3 (CLM) ran under an earlier, lighter generator variant before the v2 workload was finalized (logged in `~/.hermes/clm-bench/README.md`).

**Harness** (`~/hermes-clm/benchmarks/live_long_horizon.py`, branch `work/context-map`):
- One `hermes -p <profile> chat -c lh-bench` invocation per turn; records client wall time, reply, probe result, and the Hermes API line (`in` / `out` / `cache=reused/total`).
- Box-side per-request timing bound to each turn via **task-id watermark attribution** (procedure: `local-llm-dual-gpu-ops/references/box-log-attribution.md`); records `prompt_eval_ms`, `n_eval`, `n_gen`, `attribution_ok`.
- **Validation:** chunk-1 records re-attributed offline and cross-checked against client cache counts — **129/129 exact**; every cold re-prefill turn satisfies `n_eval == in`; remainder spot checks exact (e.g. t218: 82,762 == 82,762). Cache splits are absent from records for roughly turns ≤ 66 (capture was added mid-chunk-1); box-side `n_eval` values still confirm delta behavior there.

**Metric definitions used below:** *wire* = `in` tokens; *prefill* = box `prompt_eval_ms`; *cliff* = the 1–3 turns after a rewrite where the re-prefilled context re-caches; *cold turn* = no cache split / full re-prefill.

## 4. Results — hermes-lcm (complete)

```
turns 340 | errors 0
wire: first 32,233 → last 78,392 | max 112,608 | total 26.06 M tokens (mean 76.7 k/turn)
wall: mean 15.3 s | max 133.7 s | total 5,186 s (86.4 min)
prefill: mean 3,986 ms | total 1,355.1 s | decode mean 30.7 t/s
cache reuse: 98.7% mean on warm turns (0.9887 sum-based, over 21.9 M tokens)
probes: 39/44
```

### Wire progression (every 20 turns; both arms)

| turn | LCM in | CLM in |
|---:|---:|---:|
| 20 | 43,813 | 42,064 |
| 40 | 62,183 | 61,027 |
| 60 | 80,772 | 79,917 |
| 80 | 99,958 | 99,012 |
| 100 | 112,365 | 111,365 |
| 120 | 63,547 | — |
| 140 | 82,595 | — |
| 160 | 101,760 | — |
| 180 | 48,686 | — |
| 200 | 67,585 | — |
| 220 | 86,351 | — |
| 240 | 105,735 | — |
| 260 | 62,446 | — |
| 280 | 74,859 | — |
| 300 | 93,521 | — |
| 320 | 112,608 | — |
| 340 | 78,392 | — |

### Compactions (from the box log; each wire ~111 k → ~50–59 k)

| # | finished | between turns | engine leaf pass | messages → | tokens | DAG nodes | cliff turns (wall / pev) |
|---|---|---|---|---|---|---|---|
| C1 | 02:42:47 | 100 → 101 | 74.1 s | 221 → 33 | 24,322 → 5,021 | 2 | t101 133.7 s / 48.9 s; t102 61.5 s / 48.6 s |
| C2 | 03:06:20 | 179 → 180 | 57.6 s | 205 → 33 | 25,856 → 7,047 | 4 | t180 108.6 s / 22.3 s; t181 78.1 s / 53.0 s; t182 72.1 s / 62.4 s |
| C3 | 04:19:42 | 250 → 251 | 25.1 s | 187 → 33 | 26,293 → 7,748 | 5 | t251 87.7 s / 54.0 s; t252 68.2 s / 0.9 s |
| C4 | 04:38:42 | 320 → 321 | 36.1 s | 187 → 33 | 27,224 → 11,037 | 6 | t321 87.6 s / 36.5 s; t322 76.8 s / 67.1 s |

Plus two *cold* re-prefills not caused by compaction: t67 (harness era boundary, 104.4 s / 82.8 k tokens) and t218 (post-crash server restart, 102.2 s). Total heavy-turn prefill across the arm ≈ 599 s over ~10 turns of 340.

### Probes: 39/44 — the recall cost is localized to rewrites

| fail | asked | kind | value planted at | compaction between |
|---:|---:|---|---:|---|
| 101 | 34 turns earlier | delayed_recall | 67 | C1 |
| 108 | 34 turns earlier | delayed_recall | 74 | C1 |
| 255 | 34 turns earlier | delayed_recall | 221 | C3 |
| 262 | 34 turns earlier | delayed_recall | 228 | C3 |
| 269 | 34 turns earlier | delayed_recall | 235 | C3 |

- **5 of 13 probes that crossed a compaction missed; 0 of 31 probes that did not cross one missed.**
- Loss was lumpy across rewrites: C1 crossing probes 2/3 lost, C2 0/4, C3 3/3, C4 0/3.
- Opaque tags never failed (survived all four rewrites).

## 5. Results — hermes-clm + Context Map (complete)

**Arm completed 2026-10-14 21:43 EDT (340/340).** Two eras share this arm's history: t1–151 ran **pre-fix** (recurring full-body re-prefills; 146.9 min — the regime the fix removed), and t163–340 ran **post-fix** (frozen episode presentation). The earlier crash-stopped dataset (108/340, the 120.3 k KV-state-save crash) was superseded by this rerun and lives in git history.

```
turns 340 | errors 2 (t98 pre-fix cap-kill; t234 runaway absorbed by the 20-min cap)
wire: first 30,423 → last 65,328 | max 91,847
wall: mean 54.3 s | max 1200.0 s | total 307.9 min
      post-fix stretch t163–340: 178 turns | mean 50.9 s | 151.1 min
prefill: n=340 mean 11,013 ms | sum 3,744.6 s | decode mean 32.8 t/s
cache reuse: 0.873 sum-based
probes: 41/44
```

### What the fix changed (t163–340 regime)

- Each pressure episode costs **≤1 bounded re-prefill** (5–30 k tokens, 10–45 s) plus tail-delta calls — observed across ~20 episodes; the pre-fix arm instead chained 4–5 full-body re-prefills per episode at 70–92 k wire.
- The arm's two 20-min turns are **runaway generations** (31.5 k tokens, no EOS, unbounded `max_tokens`), absorbed unattended: the harness cap kills the client, llama-server cancels on disconnect, the next turn rebuilds (~6 min) and normal cadence resumes.
- Wire cycles a stable band (~50 → 92 k); the pre-fix 120 k crash zone was never revisited.

### Probes: 41/44

- Genuine misses: **t290, t304** (delayed-recall values).
- **t234** — timeout casualty (the turn was killed mid-generation by the cap; not a retention miss).
- Both genuine misses crossed episodes between plant and ask (t290: 4 recovery turns + 3 compacts; t304: 7 + 1) — the same rewrite-crossing failure mode as LCM's, at a lower count. CLM rewrites ~5× more often than LCM (43/44 probe windows contain ≥1 episode/compact vs LCM's 13/44), but each rewrites a small bounded span instead of the whole context — losses: 3/43 crossing windows (vs LCM 5/13); non-crossing: 0/1 (vs LCM 0/31). t234 among the 3 is the timeout casualty.

### Pre-fix era (t1–151) — kept for reference

The pre-fix rerun reproduced the mechanism the first crash-stopped arm exposed: chained recovery calls re-prefilled entire bodies at near-90 k wire (t130 window: 4 calls, 79–92 k prefills; t150–151: 69–73 k), 10 episodes ≈ 77% of run wall, monsters t98–101 (≈33 min) and t130–132 (≈27 min). Per-call forensics + fix design: `docs/recovery-episode-cost-fix.md`.

## 6. The box crash (incident report)

- **When:** ~03:44–03:57. The CLM arm's turn 108 request (~120.3 k wire — the deepest request of the run) hit a dead server. The client retried 5/5 with 62–70 s backoffs (≈ 17 min of the 1,046.1 s turn wall) before failing.
- **Cause (server log):** `E ROCm error: unspecified launch failure` in `ggml_backend_cuda_buffer_get_tensor` / `hipMemcpyAsync(..., DeviceToHost)` — i.e. during the **KV-state save to host (the `--cache-ram` state-save path)** right after the previous task released at ~120 k tokens.
- **Classification:** in-process HIP/GPU fault — *not* OOM, *not* a Windows crash (no WER/Event-Log record). n=1.
- **Recovery:** daemon `POST /start-qwen` → healthy in ~36 s; verified `/slots`: 131 k, speculative, fresh pid 69040. No data loss; all records persisted. The LCM arm was resumed afterwards and its first turn (t218) paid one expected cold re-prefill; 122 further turns ran clean.
- **Relevance:** the 120 k+ zone is where the box's KV-state save becomes a crash vector (rig-level receipt, 2026-10-14). LCM's self-bounding keeps it at ≤ ~112 k; a design without bounding runs straight into this zone.

## 7. Side-by-side

|  | hermes-lcm | hermes-clm + Map |
|---|---|---|
| Turns completed | **340 / 340** | **340 / 340** |
| Errors | 0 | 2 (t98 pre-fix cap-kill; t234 runaway absorbed by cap) |
| Wire behavior | bounded: 4 rewrites ~111 k → ~50–59 k | episode-driven cycles ~50 → 92 k; max 91,847; last 65,328 |
| Prefill / turn (mean) | **3,986 ms** | 11,013 ms (**2.76×**) |
| Wall / turn (mean) | **15.3 s** | 54.3 s (**3.55×**; post-fix stretch 50.9 s) |
| Decode | 30.7 t/s | 32.8 t/s |
| Cache reuse | 98.7% | 87.3% (episode rewrites cost parity; recovers between episodes) |
| Heavy events | 4 rewrites + 2 cold (10 heavy turns, ~599 s prefill) | ~20 episodes (≤1 bounded re-prefill each) + 2 cap-absorbed runaway generations |
| Probe recall | 39/44 (loss only at rewrite crossings: 5/13 vs 0/31) | **41/44** (2 genuine misses; 1 timeout casualty) |
| Cost model | cheap steady state; expensive, bounded, rare events; quantified recall risk at rewrites | memory-superior recall; episodic cost now bounded (≤1 re-prefill/episode) but ~3× LCM wall; levers: summary-length cap, `max_tokens` plumbing |

Both arms processed comparable wire per turn — the cost difference is **reuse efficiency** (episodes/rewrites), not workload size.

## 8. Caveats & confounds

- Single run per arm (n=1). CLM's full-run mean is inflated by the pre-fix era (t1–151) and two cap events; post-fix stretch (t163–340) mean is 50.9 s. The earlier 108-turn crash (120.3 k KV-state save) is documented in §6 and superseded by the rerun.
- The crash is rig-level (HIP state-save fault at ~120 k) — relevant to any engine that runs the lane that deep, regardless of design.
- Workload v2 audit wart: the first few turns ran under a lighter generator variant (≥ 6 LCM / ≥ 3 CLM); identical for both arms in aggregate comparison.
- Wall time includes engine aux calls and single-lane queueing; box per-task prefill is the precise instrument (validated per §3).
- Cache-split fields missing for turns ≈ 1–66 (capture added mid-run); annotations mark reconstructed values.
- CLM's engine behavior is evaluated **as-deployed** (production-soak config). A different `active_target`/recovery policy could plausibly behave differently — that would be a new arm, not this one. Final-run addendum: runaways ≈ 2 per ~250 turns (cap-absorbed unattended); t234 probe fail = timeout casualty.

## 9. Open questions / next steps

1. ~~Decide the CLM arm~~ — **Resolved:** full 340-turn arm completed post-fix.
2. ~~Pin CLM's re-prefill trigger~~ — **Resolved by the episode cost fix** (frozen presentation, tail-only divergence; forensics in `docs/recovery-episode-cost-fix.md`).
3. ~~Why CLM never rewrote~~ — **Resolved:** recovery fires; episodes now bounded (≤1 re-prefill each; see §5).
4. **Rewrite recall-loss deep dive:** which values survive C1/C3 vs C2/C4 (leaf-pass differences: 2/2/1/1 passes — no clean correlation; needs store-level analysis).
5. **Harness fix:** capture `session_id` from CLI output (parse currently returns None for both arms; cosmetic but should be exact).
6. Optional follow-ups: complete the spec's A/B/C matrix at lighter scale; re-run CLM new-arm variant; extend workload past 340 for a second cross-arm ceiling cycle.
7. **New levers (post-fix):** cap episode summary-generation length (2.5–3.2 k-token summaries cost 80–95 s of decode per episode turn); plumb `max_tokens` on box-bound requests (currently unbounded — runaways run ~17 min before the harness cap fires); port the episode freeze to the projection/native path before native-mode sessions go live.

## 10. Artifacts & receipts

- **Results (this repo):** `data/{results-lcm.jsonl, results-clm.jsonl, workload.jsonl, boxlog-lcm.txt, boxlog-clm.txt}`
- **Harness (this repo):** `benchmarks/live_long_horizon.py` (watermark-based box attribution + cache capture).
- **Attribution method:** `parse_box_window` in the harness (watermark per turn).
- **Box-side engine evidence:** LCM compaction lines in the profile agent logs; CLM `events.jsonl` per session (private state, not included).
- **Reproduce a turn:** `hermes -p bench-lcm chat -c lh-bench --oneshot -q "<task>"` (see harness `run` command for the full loop)
- **Final CLM arm + fix:** recovery-episode cost fix — design record in `docs/recovery-episode-cost-fix.md`.
- **Final report command:** `.venv/bin/python -m benchmarks.live_long_horizon report --state ~/.hermes/clm-bench/lh`
- **Runner:** the final run executed under a launchd job so it survived app restarts / reboots.
