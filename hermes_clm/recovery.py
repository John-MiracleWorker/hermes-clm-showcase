"""Host-triggered bounded recovery turn (recovery design draft, option 2).

Ownership model (docs/plans/recovery-design-draft-hermes.md):

* initiation (arming) is host/engine-owned and deterministic — it follows a
  measured exact-pressure crossing of the normal working target, never a
  voluntary model decision;
* selection stays model-owned inside ONE bounded recovery turn (restore /
  compact / archive / replace_summary via the existing ``clm_*`` verbs):
  only a *measurable* reduction below the normal working target counts as
  progress; a non-reducing edit does not;
* if the turn ends without progress, a deterministic fail-closed fallback
  archives the oldest reclaimable provenance-preserving spans (spec §19
  order, complete groups only; protected and recent turns stay intact;
  canonical history is never mutated; every archived span stays restorable).
* No chained retries: exactly one recovery turn per pressure episode. The
  turn is runtime-only state; its provenance lives in telemetry events
  (``recovery_turn`` / ``recovery_turn_settled``) and the status surface.

Activation is explicit (``CLMConfig.recovery_turn``, default off) so the
mechanism ships disabled for review and never changes baseline behavior.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# Tools whose successful use inside the turn counts as model engagement with
# recovery. Progress additionally requires a measurable token reduction.
RECOVERY_CLASS_TOOLS = frozenset({
    'clm_restore', 'clm_compact', 'clm_archive', 'clm_replace_summary',
})

# Bounded bookkeeping: the turn records at most this many observed operations.
MAX_OBSERVED_OPS = 64


def build_recovery_turn_directive(*, usable_tokens: int, attempts: int,
                                  max_attempts: int,
                                  candidates: Optional[List[Dict[str, Any]]] = None) -> str:
    """Bounded-turn directive appended to the pressure notice.

    Like every advisory, it grants no reserve, no extra attempts and no
    permission to send; the final guard still owns admission. When measured
    candidates are supplied, the largest protected-adjacent spans are named
    inline so the model can act without an extra clm_status hop.
    """
    remaining = max(0, max_attempts - attempts)
    text = (
        "CLM RECOVERY TURN (bounded — one per pressure episode)\n"
        "The measured pressure has opened this episode's single bounded recovery "
        f"turn. Reduce the working set below {usable_tokens:,} tokens in THIS turn "
        "with CLM tools (prefer clm_compact with a summary preserving exact key=value facts, identifiers, and tags plus map_path/keywords for Context Map lookup; use clm_archive only for pure filler; clm_status lists everything). "
        "Issue ONE clm_compact call immediately covering the listed UIDs below with a concise key=value summary (keep thinking brief; never guess or invent UIDs; do not call clm_status or make a second clm_compact call — after one reducing compact, answer the user turn directly). "
        "Only a measurable reduction below the normal working target counts as "
        "progress; a non-reducing edit does not. If no measurable reduction is "
        "observed by the end of this turn, CLM runs its deterministic fallback "
        "exactly once (summarized spans → old tool output → oldest non-recent "
        "messages; complete groups only; protected and recent turns stay intact; "
        "canonical history is never mutated and every archived span remains "
        f"restorable). {remaining} non-normal admission(s) remain in this episode; "
        "no further recovery turns are offered."
    )
    if candidates:
        text += ("\n\nLargest reclaimable spans measured right now "
                 "(act directly; no clm_status call needed):")
        for cand in candidates:
            members = (f" ×{cand['members']}" if int(cand.get('members') or 1) > 1 else '')
            text += (f"\n- {cand['uid']}{members} — {cand['label']}, "
                     f"~{int(cand['tokens']):,} tokens")
    return text


def _span_label(store: Any, uid: str) -> str:
    """Short human label for a reclaimable span (role + content excerpt)."""
    import re
    rec = store.get(uid)
    role = (getattr(rec, 'role', '') or 'message')
    if role == 'tool':
        role = 'tool output'
    env = getattr(rec, 'envelope', None) or {}
    content = env.get('content') if isinstance(env, dict) else None
    text = ''
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and isinstance(part.get('text'), str):
                text = part['text']
                break
    flat = ' '.join(text.split())
    excerpt = flat[:64]
    tag_match = re.search(r'\bopaque_tag_\d+\s*=\s*\S+', flat)
    if tag_match and tag_match.group(0) not in excerpt:
        excerpt = f"{excerpt} [{tag_match.group(0)}]"
    return f"{role}: {excerpt}" if excerpt else role


@dataclass
class RecoveryTurn:
    """Runtime state for the bounded recovery turn.

    ``state`` is ``idle`` or ``offered``; settlement returns to ``idle`` while
    retaining ``episode`` (monotonic count) and the last ``outcome`` for the
    status surface. Reset on session bind/reset; never persisted.
    """

    state: str = 'idle'
    episode: int = 0
    armed: Optional[Dict[str, Any]] = None
    emitted: bool = False
    ops: List[Dict[str, Any]] = field(default_factory=list)
    saw_recovery: bool = False
    outcome: Optional[Dict[str, Any]] = None
    # Pressure-episode epoch bookkeeping: at most one bounded turn per epoch.
    offered_epoch: Optional[int] = None
    used_epoch: Optional[int] = None
    # Episode-scoped render freeze: keeps directive/candidates/counter stable
    # across calls within an episode and defers mid-turn history rewrites to
    # the turn boundary so intra-turn tool loops stay warm on the prefix cache.
    frozen_directive: Optional[str] = None
    frozen_tail_echo: Optional[str] = None
    frozen_prefix: Optional[List[Dict[str, Any]]] = None
    frozen_order_len: Optional[int] = None

    def reset(self) -> None:
        self.state = 'idle'
        self.armed = None
        self.emitted = False
        self.ops = []
        self.saw_recovery = False
        self.outcome = None
        self.offered_epoch = None
        self.used_epoch = None
        self.frozen_directive = None
        self.frozen_tail_echo = None
        self.frozen_prefix = None
        self.frozen_order_len = None

    def arm(self, *, tokens: int, usable: int, active_target: int,
            attempts: int, max_attempts: int, revision: int,
            store_tokens: int, epoch: int) -> bool:
        """Open the single bounded turn for the current pressure episode.

        ``tokens`` is the measured prepared-request count that crossed the
        normal target (trigger evidence); ``store_tokens`` is the working-set
        snapshot the progress check compares against. ``epoch`` is the
        monotonic pressure-episode id: a settled turn is never re-offered for
        the same epoch (no chained retries).
        """
        if self.state != 'idle' or int(tokens) <= int(usable):
            return False
        if self.used_epoch is not None and self.used_epoch == int(epoch):
            return False
        self.episode += 1
        self.state = 'offered'
        self.offered_epoch = int(epoch)
        self.armed = {
            'episode': self.episode,
            'epoch': int(epoch),
            'tokens': int(tokens),
            'usable_tokens': int(usable),
            'active_target': int(active_target),
            'attempts': int(attempts),
            'max_attempts': int(max_attempts),
            'revision': int(revision),
            'store_tokens': int(store_tokens),
        }
        self.emitted = False
        self.ops = []
        self.saw_recovery = False
        self.frozen_directive = None
        self.frozen_tail_echo = None
        self.frozen_prefix = None
        self.frozen_order_len = None
        return True

    def mark_emitted(self) -> None:
        if self.state == 'offered':
            self.emitted = True

    def observe_op(self, *, name: str, ok: bool, working_tokens: int,
                   revision: int) -> None:
        if self.state != 'offered':
            return
        if len(self.ops) < MAX_OBSERVED_OPS:
            self.ops.append({'name': str(name), 'ok': bool(ok),
                             'working_tokens': int(working_tokens),
                             'revision': int(revision)})
        if ok and name in RECOVERY_CLASS_TOOLS:
            self.saw_recovery = True

    def settle(self, *, working_tokens: int, usable: int, trigger: str,
               fallback: Optional[Callable[[int], Optional[Dict[str, Any]]]] = None,
               ) -> Optional[Dict[str, Any]]:
        """Close the offered turn deterministically; returns the outcome record.

        ``fallback`` (when provided) executes the fail-closed deterministic
        archive to ``usable`` and returns its measurement; it runs at most once
        per episode — a settled turn never re-enters ``offered`` on its own.
        """
        if self.state != 'offered' or self.armed is None:
            return None
        armed = self.armed
        before = int(armed['store_tokens'])
        working = int(working_tokens)
        op_min = min(
            (int(op['working_tokens']) for op in self.ops
             if op.get('ok') and op.get('name') in RECOVERY_CLASS_TOOLS),
            default=working,
        )
        effective_working = min(working, op_min)
        reduced = effective_working < before
        recovered = effective_working <= int(usable)
        if recovered or (self.saw_recovery and reduced):
            selection = 'model' if (self.saw_recovery and reduced) else 'recovered'
            fallback_result = None
        elif fallback is not None:
            selection = 'fallback'
            fallback_result = fallback(int(usable))
        else:
            selection = 'fallback-disabled'
            fallback_result = None
        after = int(fallback_result['tokens_after']) if fallback_result else working
        outcome = {
            'episode': armed['episode'],
            'epoch': armed['epoch'],
            'trigger': str(trigger),
            'initiator': 'host',
            'selection': selection,
            'progress': bool(recovered or (self.saw_recovery and reduced)),
            'recovered': bool(recovered),
            'tokens_at_arm': int(armed['tokens']),
            'store_tokens_at_arm': before,
            'store_tokens_at_settle': working,
            'store_tokens_after': after,
            'usable_tokens': int(usable),
            'armed': dict(armed),
            'emitted': self.emitted,
            'saw_recovery': self.saw_recovery,
            'ops': [dict(op) for op in self.ops],
            'fallback': dict(fallback_result) if fallback_result else None,
        }
        self.outcome = outcome
        self.used_epoch = self.offered_epoch
        self.state = 'idle'
        self.armed = None
        self.emitted = False
        self.ops = []
        self.saw_recovery = False
        self.frozen_directive = None
        self.frozen_tail_echo = None
        self.frozen_prefix = None
        self.frozen_order_len = None
        return outcome

    def status(self) -> Dict[str, Any]:
        return {
            'state': self.state,
            'episode': self.episode,
            'emitted': self.emitted,
            'armed': dict(self.armed) if self.armed else None,
            'outcome': dict(self.outcome) if self.outcome else None,
        }


class RecoveryTurnMixin:
    """Engine wiring for the bounded recovery turn.

    Arming happens wherever a preparation measures pressure above the normal
    working target; the directive rides the existing pressure notice; observed
    ``clm_*`` operations are recorded through ``handle_tool_call``; settlement
    runs at the turn boundary (``on_turn_complete``) or when a later
    preparation shows the episode already recovered.
    """

    # Provided by CLMEngine.
    config: Any
    store: Any
    budget: Any
    telemetry: Any
    _pressure_episode: Any

    # ------------------------------------------------------------------ #
    # state                                                               #
    # ------------------------------------------------------------------ #

    def _initialize_recovery_turn(self) -> None:
        self._recovery_turn = RecoveryTurn()

    def _recovery_turn_enabled(self) -> bool:
        return bool(getattr(self.config, 'recovery_turn', False))

    # ------------------------------------------------------------------ #
    # initiation (host-owned)                                             #
    # ------------------------------------------------------------------ #

    def _maybe_arm_recovery_turn(self, tokens: int, usable: int, *,
                                 wire_prompt_tokens: Optional[int] = None) -> bool:
        """Arm once per pressure episode; deterministic, no model opt-in.

        ``tokens``/``usable`` are engine-currency (working-set estimate) values.
        ``wire_prompt_tokens`` is the host-reported true wire prompt size, when
        available — observability only; it never drives the crossing compare
        (the wire carries fixed overhead — system prompt + tool schemas — that
        ``clm_compact`` can never settle).
        """
        if not self._recovery_turn_enabled() or int(tokens) <= int(usable):
            return False
        turn = self._recovery_turn
        if turn.state != 'idle':
            return False
        armed = turn.arm(
            tokens=int(tokens), usable=int(usable),
            active_target=self.config.active_target_tokens,
            attempts=self._pressure_episode.attempts,
            max_attempts=self._pressure_episode.max_attempts,
            revision=self.store.revision,
            store_tokens=self.store.working_set_tokens(),
            epoch=self._pressure_episode.epoch)
        if armed:
            snapshot = turn.armed or {}
            extra = ({} if wire_prompt_tokens is None
                     else {'wire_prompt_tokens': int(wire_prompt_tokens)})
            self.telemetry.record(
                'recovery_turn', initiator='host', episode=turn.episode,
                active_tokens=int(tokens), usable_tokens=int(usable),
                active_target=self.config.active_target_tokens,
                attempts=self._pressure_episode.attempts,
                max_attempts=self._pressure_episode.max_attempts,
                store_tokens=snapshot.get('store_tokens'),
                budget_mode=str(getattr(self.config, 'budget_mode', 'approximate')),
                **extra)
        return armed

    def _observe_host_pressure(self, working_tokens: int) -> None:
        """Production (non-exact) host path: arm on the engine-currency crossing.

        The current host does not call the exact pilot-host boundary hooks; it
        re-enters ``select_context`` for every request and reports the true wire
        prompt through ``on_turn_complete(usage=...)`` (bridged into
        ``last_prompt_tokens``). The budget thresholds (``usable`` / ``hard``)
        are engine-estimated working-set currency, so arming compares the store
        working set against them — the host wire prompt additionally carries
        fixed, un-reclaimable overhead (system prompt + tool schemas, tens of
        thousands of tokens in real sessions) that ``clm_compact`` can never
        settle, so it rides arm telemetry as ``wire_prompt_tokens`` only.
        Arm exactly once per pressure episode; once pressure clears after a
        settled turn, retire the episode so a later crossing opens one fresh
        bounded turn.
        """
        if not self._recovery_turn_enabled():
            return
        usable = int(self.budget.usable_tokens)
        measured = int(working_tokens)
        turn = self._recovery_turn
        if measured > usable:
            wire = int(getattr(self, 'last_prompt_tokens', 0) or 0)
            self._maybe_arm_recovery_turn(measured, usable,
                                          wire_prompt_tokens=wire or None)
            return
        if turn.state == 'offered' or turn.used_epoch is None:
            return
        # Pressure cleared after a used turn: new episode, eligible to arm again.
        self._pressure_episode.reset()
        turn.reset()

    def _recovery_goal_met_in_turn(self) -> bool:
        turn = self._recovery_turn
        if turn.state != 'offered' or turn.armed is None or not turn.saw_recovery:
            return False
        if getattr(self.config, 'budget_mode', 'approximate') == 'exact':
            return False
        working = self.store.working_set_tokens()
        op_min = min(
            (int(op['working_tokens']) for op in turn.ops
             if op.get('ok') and op.get('name') in RECOVERY_CLASS_TOOLS),
            default=working,
        )
        effective = min(working, op_min)
        return effective <= int(turn.armed['usable_tokens']) or effective < int(turn.armed['store_tokens'])

    def _recovery_directive_body(self) -> Optional[str]:
        """Episode-frozen directive text so candidates and counters never mutate across calls."""
        turn = self._recovery_turn
        if (not self._recovery_turn_enabled() or turn.state != 'offered'
                or turn.armed is None or self._recovery_goal_met_in_turn()):
            return None
        if not turn.frozen_directive:
            turn.frozen_directive = build_recovery_turn_directive(
                usable_tokens=turn.armed['usable_tokens'],
                attempts=int(turn.armed.get('attempts', self._pressure_episode.attempts)),
                max_attempts=int(turn.armed.get('max_attempts', self._pressure_episode.max_attempts)),
                candidates=self._recovery_candidates())
        return turn.frozen_directive

    def _recovery_directive_suffix(self) -> str:
        """Directive section appended to the pressure notice while offered."""
        body = self._recovery_directive_body()
        if not body:
            return ''
        self._recovery_turn.mark_emitted()
        return '\n\n' + body

    def _recovery_candidates(self, limit: int = 8) -> List[Dict[str, Any]]:
        """Largest reclaimable active spans, in deterministic fallback order.

        Named inline in the directive so the model can act in one turn without
        an extra clm_status hop (attention sits on the tail during tool loops).
        """
        from .policies import PolicyViolation, emergency_order
        store = self.store
        out: List[Dict[str, Any]] = []
        try:
            order = emergency_order(store, self.config)
        except Exception:
            return out
        seen_groups = set()
        exact_mode = getattr(self.config, 'budget_mode', 'approximate') == 'exact'
        clm_tools = {
            'clm_compact', 'clm_archive', 'clm_status',
            'clm_replace_summary', 'clm_lookup', 'clm_restore',
        }
        primary: List[tuple[int, int, Dict[str, Any]]] = []
        secondary: List[tuple[int, int, Dict[str, Any]]] = []
        for idx, uid in enumerate(order):
            rec = store.get(uid)
            if rec is None or not rec.active:
                continue
            try:
                members = store.resolve_groups([uid])
            except PolicyViolation:
                continue
            key = tuple(members)
            if key in seen_groups or any(not store.get(u).active for u in members):
                continue
            seen_groups.add(key)
            tokens = sum(int(store.get(u).token_count or 0) for u in members)
            entry = {
                'uid': members[0], 'members': len(members), 'tokens': tokens,
                'label': _span_label(store, members[0]),
            }
            if exact_mode:
                primary.append((-tokens, idx, entry))
                continue
            first = store.get(members[0])
            env = getattr(first, 'envelope', None) or {}
            tcs = env.get('tool_calls') or []
            names = {tc.get('function', {}).get('name') for tc in tcs if isinstance(tc, dict)}
            is_clm_group = bool(names and names.issubset(clm_tools))
            is_short_ack = (len(members) == 1 and getattr(first, 'role', '') == 'assistant'
                            and not tcs and tokens < 80)
            if is_clm_group or is_short_ack:
                secondary.append((-tokens, idx, entry))
            elif tokens >= 400:
                primary.append((0, -tokens, idx, entry))  # type: ignore[arg-type]
            elif getattr(first, 'role', '') == 'user':
                # Raw user notes: oldest first so the earliest notes are
                # always offered ahead of recent small spans; emergency_order
                # protects the recent recall-probe window separately.
                primary.append((1, idx, -tokens, entry))  # type: ignore[arg-type]
            elif tokens < 200 and (getattr(first, 'role', '') == 'tool' or tcs):
                # Tiny non-user tool groups: low reclaim value; demote
                # alongside CLM scaffolding and short assistant acks.
                secondary.append((-tokens, idx, entry))
            else:
                primary.append((1, idx, -tokens, entry))  # type: ignore[arg-type]
        if exact_mode:
            primary.sort(key=lambda item: (item[0], item[1]))
            return [item[2] for item in primary[:limit]]
        primary.sort(key=lambda item: (item[0], item[1], item[2]))  # type: ignore[index]
        secondary.sort(key=lambda item: (item[0], item[1]))
        pool = [item[-1] for item in primary] + [item[-1] for item in secondary]
        chosen = pool[:limit]
        if limit == 8 and sum(int(c['tokens']) for c in chosen) < 3000 and len(pool) > limit:
            chosen = pool[:min(14, len(pool))]
        return chosen

    def _recovery_tail_echo(self) -> Optional[str]:
        """Tail-visible copy of the directive (attention-sink mitigation).

        The prefix notice sits far above an active tool loop; while the bounded
        turn is offered the same directive is echoed after the latest result.
        """
        turn = self._recovery_turn
        if (not self._recovery_turn_enabled() or turn.state != 'offered'
                or turn.armed is None or self._recovery_goal_met_in_turn()):
            return None
        if not turn.frozen_tail_echo:
            body = self._recovery_directive_body()
            if not body:
                return None
            turn.frozen_tail_echo = (
                "CLM RECOVERY TURN — tail reminder, host-injected "
                "(read this before your next action; the same directive also "
                "appears in the pressure notice):\n" + body
            )
        turn.mark_emitted()
        return turn.frozen_tail_echo

    def _retire_recovery_scaffolding(self) -> List[str]:
        """Archive unprotected CLM management tool groups and orphaned assistant acks.

        Runs only on the approximate non-native host path during/after a recovery
        turn so CLM's own tool-call JSON and orphaned acknowledgements do not
        re-inflate the working set or trigger back-to-back episodes. Never
        touches user messages or protected turns.
        """
        if (not self._recovery_turn_enabled()
                or getattr(self.config, 'budget_mode', 'approximate') != 'approximate'
                or getattr(self, '_origin_mode', 'standalone') == 'native'):
            return []
        from .policies import PolicyViolation
        store = self.store
        to_archive: List[str] = []
        clm_tools = {
            'clm_compact', 'clm_archive', 'clm_status',
            'clm_replace_summary', 'clm_lookup', 'clm_restore',
        }
        long_horizon = len(store.order) > 40
        cutoff = max(0, len(store.order) - 20)
        try:
            for g in store.groups():
                if not g.complete:
                    continue
                recs = [store.messages[u] for u in g.source_uids]
                if any(r.protected or not r.active for r in recs):
                    continue
                asst_env = store.messages[g.assistant_uid].envelope
                tcs = asst_env.get('tool_calls') or []
                names = {tc.get('function', {}).get('name') for tc in tcs if isinstance(tc, dict)}
                group_tokens = sum(int(r.token_count or 0) for r in recs)
                if names and names.issubset(clm_tools):
                    to_archive.extend(g.source_uids)
                elif long_horizon and group_tokens < 200 and max(r.index for r in recs) < cutoff:
                    to_archive.extend(g.source_uids)
            for i, uid in enumerate(store.order):
                rec = store.messages[uid]
                if not rec.active or rec.protected or uid in to_archive:
                    continue
                if rec.role == 'assistant' and not rec.envelope.get('tool_calls') and int(rec.token_count or 0) < 80:
                    prev_user = None
                    for j in range(i - 1, -1, -1):
                        cand = store.messages[store.order[j]]
                        if cand.role == 'user':
                            prev_user = cand
                            break
                    if (prev_user is not None and not prev_user.active) or (long_horizon and rec.index < cutoff):
                        to_archive.append(uid)
            if to_archive:
                resolved = store.resolve_groups(to_archive)
                eligible = [u for u in resolved if store.get(u) is not None and store.get(u).active and not store.get(u).protected]
                if eligible:
                    store.mark_archived(eligible)
                    return eligible
        except (PolicyViolation, Exception):
            return []
        return []

    # ------------------------------------------------------------------ #
    # observation + settlement                                            #
    # ------------------------------------------------------------------ #

    def _observe_recovery_tool(self, name: str, result: str) -> None:
        """Record a dispatched context-tool call while the turn is offered."""
        if (not self._recovery_turn_enabled()
                or self._recovery_turn.state != 'offered'):
            return
        try:
            ok = bool(json.loads(result).get('ok'))
        except Exception:
            ok = False
        self._recovery_turn.observe_op(
            name=name, ok=ok, working_tokens=self.store.working_set_tokens(),
            revision=self.store.revision)

    def _settle_recovery_turn(self, *, trigger: str) -> Optional[Dict[str, Any]]:
        """Settle the offered turn; runs the fail-closed fallback without progress."""
        if (not self._recovery_turn_enabled()
                or self._recovery_turn.state != 'offered'):
            return None
        fallback = (self._recovery_fallback_archive
                    if self.config.emergency_host_fallback else None)
        outcome = self._recovery_turn.settle(
            working_tokens=self.store.working_set_tokens(),
            usable=self.budget.usable_tokens,
            trigger=trigger, fallback=fallback)
        if outcome is None:
            return None
        fallback = outcome.get('fallback') or {}
        # The directive and its tail echo are valid only while offered; retire
        # them atomically so no later render repeats a settled instruction.
        notice = getattr(self, '_exact_pressure_notice', None)
        if notice:
            text = notice.get('text') or ''
            marker = '\n\nCLM RECOVERY TURN'
            if marker in text:
                notice['text'] = text.split(marker, 1)[0]
            notice['recovery_suffix'] = ''
            notice['tail_echo'] = ''
        self.telemetry.record(
            'recovery_turn_settled', initiator='host',
            episode=outcome['episode'], trigger=outcome['trigger'],
            selection=outcome['selection'], progress=outcome['progress'],
            recovered=outcome['recovered'],
            store_tokens_at_arm=outcome['store_tokens_at_arm'],
            store_tokens_at_settle=outcome['store_tokens_at_settle'],
            store_tokens_after=outcome['store_tokens_after'],
            usable_tokens=outcome['usable_tokens'],
            observed_ops=[op['name'] for op in outcome['ops']],
            saw_recovery=outcome['saw_recovery'],
            archived=list(fallback.get('removed') or []),
            restore_point=fallback.get('restore_point'))
        return outcome

    # ------------------------------------------------------------------ #
    # deterministic fail-closed fallback (spec §19 order)                 #
    # ------------------------------------------------------------------ #

    def _recovery_fallback_archive(self, target_tokens:
                                   int) -> Optional[Dict[str, Any]]:
        """Archive complete reclaimable groups until the working set is at or
        below ``target_tokens``. Mirrors the emergency fallback's store-level
        discipline inside its own transaction; canonical history is untouched
        and every archived span stays restorable via ``clm_restore``."""
        from .policies import PolicyViolation, emergency_order
        snapshot = self._begin_transaction()
        revision_before = self.store.revision
        try:
            tokens_before = self.store.working_set_tokens()
            order = emergency_order(self.store, self.config)
            removed: List[str] = []
            headroom = tokens_before - int(target_tokens)
            # One accepted-frame check owns the complete compound; the nested
            # mutations skip per-call exact checks and the ``_persist`` recount
            # validates the final state.
            with self._exact_edit_scope('recovery_fallback_archive'):
                for uid in order:
                    if headroom <= 0:
                        break
                    rec = self.store.get(uid)
                    if rec is None or not rec.active or uid in removed:
                        continue
                    try:
                        resolved = self.store.resolve_groups([uid])
                        # All members must be eligible; recent/tool closure is atomic.
                        if any(self.store.get(u).protected or u not in order
                               for u in resolved):
                            continue
                        self.store.mark_archived(resolved)
                    except PolicyViolation:
                        continue
                    removed.extend(resolved)
                    headroom -= sum(self.store.get(u).token_count or 0
                                    for u in resolved)
                if removed:
                    self.store.bump_revision()
            tokens_after = self.store.working_set_tokens()
            result = {
                'removed': removed,
                'tokens_before': tokens_before,
                'tokens_after': tokens_after,
                'saved': max(0, tokens_before - tokens_after),
                'below_target': tokens_after <= int(target_tokens),
                'restore_point': {
                    'archived_uids': list(removed),
                    'revision_before': revision_before,
                    'revision_after': self.store.revision,
                    'recoverable_via': 'clm_restore',
                },
            }
            self._persist()
            self._publish_context_transaction()
            return result
        except BaseException:
            self._rollback_transaction(snapshot)
            raise
