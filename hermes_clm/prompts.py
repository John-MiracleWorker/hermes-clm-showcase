"""Model-facing text (spec §15, §17).

The context-management guidance belongs in the protected system layer —
external content (tool output, retrieved docs) can never redefine CLM policy
(spec §32.5).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

CLM_SYSTEM_GUIDANCE = """You can manage your own active working context using CLM tools.

Your goal is task success, not maximum compression.

Preserve:
- user requirements,
- architectural decisions,
- unresolved work,
- important observations,
- tool outputs that may matter later,
- reasons behind non-obvious decisions.

Archive or compact:
- superseded plans,
- dead ends,
- repetitive tool output,
- logs that have already been analyzed,
- details whose conclusions can be preserved more compactly.

Do not compact merely because a threshold was crossed.
Compact when doing so improves your ability to continue the task.

Archived context can be restored later (clm_restore). Inspect state with clm_status
before deciding. Summaries must preserve objectives, constraints, decisions,
rejected approaches and why, identifiers, filenames, commands, errors, measured
results, unresolved questions, and next actions."""


def _notice_header(active: int, usable: int, pct: int) -> str:
    return (
        "CLM context status:\n"
        f"{active:,} / {usable:,} working tokens are active ({pct}%)."
    )


def build_notice(level: float, active_tokens: int, usable_tokens: int) -> str:
    """Escalating nudge notices (spec §17)."""
    pct = round(active_tokens / usable_tokens * 100) if usable_tokens else 0
    header = _notice_header(active_tokens, usable_tokens, pct)
    if level <= 0.25:
        return header + "\n\nContext usage is low. No action needed."
    if level <= 0.50:
        return (
            header
            + "\n\nYou have substantial headroom.\n"
            "Compact only if stale or low-value context is interfering with the task."
        )
    if level <= 0.75:
        return (
            header
            + "\n\nContext is filling up. Consider proactive cleanup: compact stale, "
            "superseded, or analyzed tool output into summaries, and archive dead ends. "
            "Keep active constraints, decisions, and unresolved work."
        )
    return (
        header
        + "\n\nContext is near capacity.\n"
        "Before continuing substantial work, compact or archive stale context while "
        "preserving active constraints, decisions, and unresolved work. "
        "Use clm_status to see the budget, then act."
    )


def build_exact_notice(level: float, verified_tokens: int, usable_tokens: int) -> str:
    """Use a prior verified task frame for advice, never current admission."""
    advice = build_notice(level, verified_tokens, usable_tokens).split("\n\n", 1)[1]
    return (
        "CLM context status (previous exact task frame):\n"
        f"The previous verified rendered task prompt used {verified_tokens:,} tokens; "
        f"the normal working target is {usable_tokens:,} tokens.\n\n"
        + advice
        + "\n\nThis is a prior exact count, not a count of the current request. "
        "The final dispatch guard counts the complete current payload, including this notice."
    )


def build_prepared_notice(level: float, usable_tokens: int) -> str:
    """Stable pressure-band advice from this complete request's preparation.

    Refreshing the notice can change token counts. Describe a measured band
    during preparation, not an exact final count or permission to dispatch.
    """
    advice = build_notice(level, 0, usable_tokens).split("\n\n", 1)[1]
    return (
        "CLM context status (current request preparation):\n"
        f"A complete rendering during this preparation reached at least {level:.0%} "
        f"of the normal working target of {usable_tokens:,} tokens.\n\n"
        + advice
        + "\n\nPreparation recounts notice and Context Map changes. "
        "The separate final guard counts the complete final payload. "
        "This advisory grants no reserve, extra attempts, or permission to send."
    )


def build_exact_recovery_notice(*, usable_tokens: int, active_target: int,
                                above_target: bool, attempts: int,
                                max_attempts: int, nonprogress: int,
                                max_nonprogress: int) -> str:
    """Describe measured pressure and existing limits without granting admission."""
    pressure = (
        f"A complete rendering during this preparation exceeded the operating cap of "
        f"{active_target:,} tokens and the normal working target of {usable_tokens:,} tokens. "
        "It requires explicitly classified temporary provider headroom."
        if above_target else
        f"A complete rendering during this preparation exceeded the normal working target of "
        f"{usable_tokens:,} tokens. The internal reserve ends at {active_target:,} tokens; "
        "using it requires a bounded pending-group completion or restore reason."
    )
    remaining = max(0, max_attempts - attempts)
    return (
        "CLM context status (current request preparation):\n"
        "CLM RECOVERY REQUIRED\n" + pressure
        + f"\n\nBefore this proposed request, {attempts} of {max_attempts} non-normal requests "
        f"were already admitted in this pressure episode; {remaining} admission(s) remain "
        "before this proposed request. This request, if admitted above the normal target, "
        "counts against that allowance. "
        f"Non-progressing edits: {nonprogress} of {max_nonprogress}. "
        "The guard refuses further non-normal requests when either limit is exhausted; "
        "no automatic compression is promised."
        + f"\n\nRecover below {usable_tokens:,} tokens before more task expansion. "
        "Use CLM tools to summarize analyzed output or archive superseded context, "
        "preserving active constraints, decisions, findings and unresolved work. "
        "If identifiers are needed, inspect clm_status. Reading, searching or testing "
        "without reducing context does not replenish the recovery allowance."
        + "\n\nPreparation recounts notice and Context Map changes. "
        "The separate final guard counts the complete final payload. "
        "This advisory grants no reserve, extra attempts, or permission to send."
    )


def urgent_overflow_notice(active_tokens: int, usable_tokens: int,
                           retries_left: int) -> str:
    """Spec §18 step 3: inject an urgent CLM-management turn on overflow."""
    pct = round(active_tokens / usable_tokens * 100) if usable_tokens else 0
    return (
        "CLM OVERFLOW: the working context cannot accept the next expansion "
        f"({active_tokens:,} / {usable_tokens:,} tokens, {pct}%). "
        f"Compact or archive stale context NOW to free space. {retries_left} "
        "compaction retry(ies) remain before host emergency compression runs. "
        "Preserve protected instructions, the root task, decisions, and unresolved work."
    )


def emergency_summary_prompt(transcript_excerpt: str) -> str:
    """Host-side emergency compression prompt (spec §19, step 6)."""
    return (
        "You are the host emergency compressor for a Hermes CLM context engine. "
        "The model failed to free context in time. Compress the following transcript "
        "span into a single structured summary preserving: objective, constraints, "
        "decisions, rejected approaches and why, identifiers, filenames, commands, "
        "API names, errors, measured results, unresolved questions, next actions.\n\n"
        f"TRANSCRIPT:\n{transcript_excerpt}"
    )


def toolobs_placeholder(ref: str, head: str, tail: str, total_chars: int,
                        window_chars: int) -> str:
    """Spec §27: offloaded tool output renders as an artifact reference."""
    return (
        f"Tool output stored as artifact {ref}.\n"
        f"Showing first {window_chars:,} and last {window_chars:,} characters "
        f"(total {total_chars:,}).\n"
        f"--- BEGIN ({ref}) ---\n{head}\n--- END HEAD ---\n"
        f"--- BEGIN ({ref}) ---\n{tail}\n--- END TAIL ---"
    )


def render_summary_message(summary: Any, index: int) -> Dict[str, Any]:
    """Render a synthetic summary as an OpenAI-format message (spec §11 step 6:
    the summary takes the logical position of the removed span)."""
    return {
        "role": "user",
        "content": (
            f"[CLM SUMMARY {summary.uid} rev {summary.revision}; "
            f"archived membership: {len(summary.source_message_ids)} messages; "
            f"known inputs: {len(summary.derivation_source_uids or [])} originals "
            f"(joins in clm_status); restore membership: clm_restore "
            f"summary_id={summary.uid}]\n{summary.content}"
        ),
        "_clm_summary": summary.uid,
        "_clm_anchor_index": index,
    }
