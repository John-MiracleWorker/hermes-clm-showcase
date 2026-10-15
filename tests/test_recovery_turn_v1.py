"""Bounded recovery turn (recovery-design option 2) — offline fixture.

The fixture distinguishes *progress* (a measurable reduction below the
normal working target, by the model or by the deterministic fail-closed
fallback) from *repeated investigation* (observed activity that does not
reduce the working set). Deterministic: the admission counter is a fixed
stub; all store measurements are real.

Ownership pinned by these tests:
* arming is host/engine-owned and follows measured pressure (one turn per
  pressure episode — no chained retries);
* selection inside the turn is model-owned; only measurable reduction counts;
* non-action / non-reducing activity settles once, with the fallback
  archiving complete reclaimable groups to below target, reversible and
  provenance-recorded;
* with the feature flag off, baseline advisory-only behavior is unchanged.
"""
from copy import deepcopy
import json

import pytest

from hermes_clm.recovery import RecoveryTurn
from test_exact_atomic_v6 import setup
from test_exact_pending_group_reserve import group, prepare, request_for
from test_exact_pressure_notices import notices


# --------------------------------------------------------------------------- #
# unit: turn lifecycle                                                        #
# --------------------------------------------------------------------------- #

def test_turn_arms_once_per_episode_and_settles_once():
    turn = RecoveryTurn()

    def boom(_target):
        raise AssertionError('fallback must not run for model progress')

    assert turn.arm(tokens=30000, usable=28672, active_target=32768,
                    attempts=0, max_attempts=6, revision=1, store_tokens=31000,
                    epoch=3) is True
    assert turn.state == 'offered' and turn.episode == 1
    # A second arm while offered is refused (one bounded turn at a time).
    assert turn.arm(tokens=30000, usable=28672, active_target=32768,
                    attempts=0, max_attempts=6, revision=1, store_tokens=31000,
                    epoch=3) is False

    turn.observe_op(name='clm_archive', ok=True, working_tokens=30000, revision=2)
    outcome = turn.settle(working_tokens=30000, usable=28672,
                          trigger='turn_complete', fallback=boom)
    assert outcome['selection'] == 'model' and outcome['progress'] is True
    assert outcome['fallback'] is None
    assert outcome['epoch'] == 3
    # Settled turns never re-settle; the fallback ran zero times.
    assert turn.settle(working_tokens=30000, usable=28672,
                       trigger='turn_complete', fallback=boom) is None

    # Same episode epoch can never re-arm (no chained retries)...
    assert turn.arm(tokens=31000, usable=28672, active_target=32768,
                    attempts=1, max_attempts=6, revision=5, store_tokens=31000,
                    epoch=3) is False
    # ...but a new (reset) episode epoch can.
    assert turn.arm(tokens=31000, usable=28672, active_target=32768,
                    attempts=0, max_attempts=6, revision=6, store_tokens=31000,
                    epoch=4) is True
    assert turn.episode == 2


def test_non_action_settles_with_fallback_exactly_once():
    turn = RecoveryTurn()
    calls = []

    def fallback(target):
        calls.append(target)
        return {'removed': ['u1'], 'tokens_before': 31000, 'tokens_after': 28000,
                'saved': 3000, 'below_target': True,
                'restore_point': {'archived_uids': ['u1']}}

    assert turn.arm(tokens=31000, usable=28672, active_target=32768,
                    attempts=2, max_attempts=6, revision=1, store_tokens=31000,
                    epoch=7) is True
    outcome = turn.settle(working_tokens=31000, usable=28672,
                          trigger='turn_complete', fallback=fallback)
    assert calls == [28672]
    assert outcome['selection'] == 'fallback' and outcome['progress'] is False
    assert outcome['store_tokens_after'] == 28000
    assert outcome['fallback']['below_target'] is True
    # No chained retries: nothing left to settle, fallback not re-invoked.
    turn.settle(working_tokens=31000, usable=28672, trigger='turn_complete',
                fallback=fallback)
    assert calls == [28672]


def test_settle_without_fallback_marks_fallback_disabled():
    turn = RecoveryTurn()
    turn.arm(tokens=31000, usable=28672, active_target=32768, attempts=0,
             max_attempts=6, revision=1, store_tokens=31000, epoch=1)
    outcome = turn.settle(working_tokens=31000, usable=28672,
                          trigger='turn_complete', fallback=None)
    assert outcome['selection'] == 'fallback-disabled'
    assert outcome['fallback'] is None


# --------------------------------------------------------------------------- #
# engine fixture helpers                                                      #
# --------------------------------------------------------------------------- #

def big_group(number=0, body_chars=2600):
    ids = [f'big-{number}-a', f'big-{number}-b']
    return [{'role': 'assistant', 'content': '', 'tool_calls': [
        {'id': cid, 'type': 'function', 'function': {'name': 'read_file', 'arguments': '{}'}}
        for cid in ids]}, *[{'role': 'tool', 'tool_call_id': cid,
                            'content': 'S' * body_chars} for cid in ids]]


def armed_engine(tmp_path, margin_groups=6):
    """Engine with recovery turn enabled, pressure armed, the current
    presentation accepted as the counted frame, and a working set large enough
    for the fallback to reclaim."""
    engine, counter, source = prepare(tmp_path)
    engine.config.recovery_turn = True
    engine.config.protect_latest_assistant_turns = 1
    sample = big_group(9000)
    per_group = max(1, engine.counter.count_messages(sample))
    need = engine.budget.usable_tokens + per_group * margin_groups
    groups = (need // per_group) + 2
    for n in range(groups):
        source.extend(big_group(n))
    engine.select_context(source, conversation_messages=source)
    assert engine.store.working_set_tokens() > engine.budget.usable_tokens

    source.extend(group(999))
    # Measured pressure crosses the normal target: the bounded turn is armed and
    # its directive rides the prepared request.
    counter.tokens = 32769
    route = engine._budget_frame['route']
    prepared = engine.prepare_context_request(request_for(engine, source), route=route)
    assert engine._recovery_turn.state == 'offered'
    # The host accepts the prepared presentation before dispatch (fixture stub:
    # a post-recovery normal count); engine-initiated edits are gated on the
    # accepted presentation matching the store.
    counter.tokens = 28000
    decision = engine.check_final_request(prepared, route=route)
    assert decision['allowed'], decision
    assert engine._budget_frame['source_uids'] == engine.store.order
    return engine, counter, source, prepared


# --------------------------------------------------------------------------- #
# integration: initiation, directive, settlement                              #
# --------------------------------------------------------------------------- #

def test_pressure_crossing_arms_one_bounded_turn_and_directs_the_notice(tmp_path):
    engine, counter, source = prepare(tmp_path)
    engine.config.recovery_turn = True
    engine.config.protect_latest_assistant_turns = 1
    source.extend(group())
    counter.tokens = 32769
    route = engine._budget_frame['route']
    prepared = engine.prepare_context_request(request_for(engine, source), route=route)

    text = notices(prepared)[0]['content']
    assert 'CLM RECOVERY REQUIRED' in text          # the measured advisory stays
    assert 'CLM RECOVERY TURN (bounded' in text      # this episode's single turn
    assert 'no further recovery turns are offered' in text
    # Attention-sink mitigation: the directive is echoed after the latest result.
    tail = prepared['messages'][-1]
    assert tail['role'] == 'system'
    assert tail['content'].startswith('CLM RECOVERY TURN — tail reminder')

    turn = engine._recovery_turn
    assert turn.state == 'offered' and turn.episode == 1 and turn.emitted
    assert turn.armed['tokens'] == 32769
    assert turn.armed['usable_tokens'] == 28672
    events = [e for e in engine.telemetry.recent(100) if e.get('event') == 'recovery_turn']
    assert len(events) == 1 and events[0]['initiator'] == 'host'

    # Preparation grants nothing: the final guard still owns admission.
    assert engine.check_final_request(prepared, route=route)['allowed']
    # The directive does not consume or extend the episode's allowance.
    assert engine._pressure_episode.attempts == 1


def test_default_flag_preserves_advisory_only_behavior(tmp_path):
    engine, counter, source = prepare(tmp_path)
    assert engine.config.recovery_turn is False
    engine.config.protect_latest_assistant_turns = 1
    source.extend(group())
    counter.tokens = 32769
    prepared = engine.prepare_context_request(
        request_for(engine, source), route=engine._budget_frame['route'])
    text = notices(prepared)[0]['content']
    assert 'CLM RECOVERY REQUIRED' in text
    assert 'CLM RECOVERY TURN' not in text
    assert all('CLM RECOVERY TURN' not in (m.get('content') or '')
               for m in prepared['messages'])
    assert engine._recovery_turn.state == 'idle'
    engine.on_turn_complete(source)
    assert engine._recovery_turn.outcome is None
    assert 'recovery_turn' not in engine.get_status()


def test_model_selected_reduction_settles_as_progress_without_fallback(tmp_path):
    engine, counter, source, prepared = armed_engine(tmp_path)
    before = engine.store.working_set_tokens()

    victims = [uid for uid in engine.store.order
               if engine.store.get(uid).role == 'tool'][:6]
    # Fixture convention: the stub reports the post-edit recount the guard
    # validates (a real counter would measure the archived presentation).
    counter.tokens = 28000
    result = json.loads(engine.handle_tool_call(
        'clm_archive', {'message_ids': victims, 'reason': 'recovery turn selection'}))
    assert result['ok'], result
    assert engine._recovery_turn.saw_recovery is True
    assert engine.store.working_set_tokens() < before

    engine.on_turn_complete(source)
    outcome = engine._recovery_turn.outcome
    assert outcome['selection'] == 'model' and outcome['progress'] is True
    assert outcome['fallback'] is None
    assert outcome['trigger'] == 'turn_complete'
    settled = [e for e in engine.telemetry.recent(200)
               if e.get('event') == 'recovery_turn_settled']
    assert len(settled) == 1 and settled[0]['selection'] == 'model'


def test_activity_without_reduction_is_not_progress_and_falls_back_once(tmp_path):
    engine, counter, source, prepared = armed_engine(tmp_path)
    before = engine.store.working_set_tokens()

    # Observed activity inside the turn that does not reduce the working set:
    # repeated investigation is not progress.
    assert json.loads(engine.handle_tool_call('clm_status', {}))['ok'] is True
    assert engine._recovery_turn.ops and engine._recovery_turn.saw_recovery is False
    assert engine.store.working_set_tokens() == before

    # Fixture convention: the stub reports the post-fallback recount the guard
    # validates when the deterministic archive settles at the turn boundary.
    counter.tokens = 28000
    engine.on_turn_complete(source)
    outcome = engine._recovery_turn.outcome
    assert outcome['selection'] == 'fallback'
    assert outcome['progress'] is False
    fallback = outcome['fallback']
    assert fallback['removed'] and fallback['saved'] > 0
    assert fallback['below_target'] is True
    assert engine.store.working_set_tokens() <= engine.budget.usable_tokens
    assert fallback['restore_point']['archived_uids'] == fallback['removed']

    # Bound: exactly one settlement; further boundaries and further pressure
    # in the same episode never re-offer or re-run anything.
    engine.on_turn_complete(source)
    prepared2 = engine.prepare_context_request(
        request_for(engine, source), route=engine._budget_frame['route'])
    assert engine._recovery_turn.state == 'idle'
    assert 'CLM RECOVERY TURN (bounded' not in notices(prepared2)[0]['content']
    settled = [e for e in engine.telemetry.recent(200)
               if e.get('event') == 'recovery_turn_settled']
    assert len(settled) == 1


def test_fallback_archives_complete_groups_and_stays_reversible(tmp_path):
    engine, counter, source, prepared = armed_engine(tmp_path)
    canonical_uids = list(engine.store.order)
    canonical_env = deepcopy([engine.store.messages[uid].envelope for uid in engine.store.order])
    calls_of = {uid: {c.get('id') for c in (engine.store.messages[uid].envelope.get('tool_calls') or [])}
                for uid in canonical_uids}
    parents = {cid: uid for uid, ids in calls_of.items() for cid in ids}

    counter.tokens = 28000
    engine.on_turn_complete(source)
    removed = set(engine._recovery_turn.outcome['fallback']['removed'])
    assert removed

    # Canonical history untouched: same order, byte-identical envelopes.
    assert list(engine.store.order) == canonical_uids
    assert [engine.store.messages[uid].envelope for uid in engine.store.order] == canonical_env

    # Complete tool groups only: no orphaned parents or results.
    for uid, envelope in zip(canonical_uids, canonical_env):
        if envelope.get('role') == 'tool':
            parent = parents.get(envelope.get('tool_call_id'))
            assert (uid in removed) == (parent in removed), 'partial tool group archived'
        for cid in calls_of[uid]:
            assert (uid in removed) == (parents[cid] in removed), 'partial tool group archived'

    # Reversible with provenance: the restore point names the span, and
    # clm_restore brings an archived group back.
    assert set(engine._recovery_turn.outcome['fallback']['restore_point']['archived_uids']) == removed
    # Re-accept the current presentation as the counted frame (as the host does
    # before each dispatch) so the restore edit passes the exact-mode guard.
    counter.tokens = 28000
    route = engine._budget_frame['route']
    frame_request = request_for(engine, source)
    assert engine.check_final_request(frame_request, route=route)['allowed']
    first = sorted(removed, key=canonical_uids.index)[0]
    group_uids = engine.store.resolve_groups([first])
    # Post-restore recount convention; restoring uses the reserve class with
    # the 'restore' reason.
    counter.tokens = 30000
    restored = json.loads(engine.handle_tool_call('clm_restore', {'message_ids': group_uids}))
    assert restored['ok'], restored
    assert all(engine.store.get(u).active for u in group_uids)


def test_directive_names_reclaimable_spans_inline_and_at_tail(tmp_path):
    engine, counter, source, prepared = armed_engine(tmp_path)

    # Inline actionability: the notice names the largest reclaimable spans with
    # sizes, so no clm_status hop is needed before a reducing edit.
    text = notices(prepared)[0]['content']
    assert 'Largest reclaimable spans measured right now' in text
    assert 'no clm_status call needed' in text
    listed = [line for line in text.splitlines() if line.startswith('- msg')]
    assert listed and all('tokens' in line for line in listed)

    # Tail visibility: the same directive is echoed after the latest result.
    tail = prepared['messages'][-1]
    assert tail['role'] == 'system'
    assert tail['content'].startswith('CLM RECOVERY TURN — tail reminder')
    assert listed[0] in tail['content']
    # The accepted frame carries the same tail echo (render/frame coherent).
    assert engine._budget_frame['tail_echo'] == tail['content']

    # Settlement retires the directive and the echo: no later render repeats it.
    counter.tokens = 28000
    engine.on_turn_complete(source)
    assert engine._recovery_turn.state == 'idle'
    assert not (engine._exact_pressure_notice or {}).get('tail_echo')
    rendered = engine._render_budget_messages(engine.store)
    assert all('CLM RECOVERY TURN' not in (m.get('content') or '')
               for m in rendered if isinstance(m, dict))
