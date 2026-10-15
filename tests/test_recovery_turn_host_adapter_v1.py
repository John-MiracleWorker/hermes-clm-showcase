"""Production-host adapter for the bounded recovery turn (engine-currency arming).

The current Hermes host does not call the exact pilot-host boundary hooks
(``prepare_context_request`` / ``check_final_request`` / ``bind_exact_counter``);
per request it calls ``select_context``, and per finished turn it calls
``on_turn_complete(messages, usage=...)`` with the true provider usage. These
tests pin the adapter that arms the bounded turn from the ENGINE-currency
working-set crossing (the same currency as the ``usable`` / ``hard`` budget
thresholds) and attaches the directive + tail echo through the non-exact
render path (the path production actually uses).

The host-reported wire prompt (``usage.prompt_tokens`` → ``last_prompt_tokens``)
additionally carries fixed, un-reclaimable overhead — system prompt + tool
schemas (tens of thousands of tokens in real sessions) that ``clm_compact``
can never settle — so it must NOT drive the crossing compare; it rides arm
telemetry as ``wire_prompt_tokens`` for observability only.
"""

import json

from hermes_clm.config import CLMConfig
from hermes_clm.engine import CLMEngine


def big_group(number=0, body_chars=2600):
    ids = [f'big-{number}-a', f'big-{number}-b']
    return [{'role': 'assistant', 'content': '', 'tool_calls': [
        {'id': cid, 'type': 'function', 'function': {'name': 'read_file', 'arguments': '{}'}}
        for cid in ids]}, *[{'role': 'tool', 'tool_call_id': cid,
                            'content': 'S' * body_chars} for cid in ids]]


def make_engine(tmp_path, groups=4, **overrides):
    cfg = dict(recovery_turn=True, persistence=False, event_log=False,
               protect_latest_user_turns=0, protect_latest_assistant_turns=0)
    cfg.update(overrides)
    engine = CLMEngine(CLMConfig(**cfg))
    engine.on_session_start('adapter-test')
    source = [{'role': 'user', 'content': 'root task'}]
    for n in range(groups):
        source.extend(big_group(n))
    engine.on_turn_complete(source)  # full history; no usage yet
    return engine, source


def cross(engine, source, margin=3500):
    """Grow the store working set clearly past ``usable`` (+margin).

    Margin stays below ``hard_limit`` (32,768 on the default active_target
    32,768 / reserve 4,096) so the select path does not enter the overflow
    regime; a single later archive (~2.4k estimated) still cannot dip the
    working set back under ``usable``.
    """
    n = 100
    for m in source:
        if not (isinstance(m, dict) and m.get('role') == 'assistant'):
            continue
        for tc in (m.get('tool_calls') or []):
            cid = (tc or {}).get('id') if isinstance(tc, dict) else None
            if isinstance(cid, str) and cid.startswith('big-'):
                try:
                    n = max(n, int(cid.split('-')[1]) + 1)
                except (ValueError, IndexError):
                    pass
    while engine.store.working_set_tokens() <= int(engine.budget.usable_tokens) + margin:
        source.extend(big_group(n))
        n += 1
        engine.on_turn_complete(source)
        if n > 300:
            raise AssertionError('could not cross usable')
    return source


def reported(tokens):
    return {'prompt_tokens': tokens, 'completion_tokens': 5, 'total_tokens': tokens + 5}


def directive_in(messages):
    return any(isinstance(m.get('content'), str) and 'CLM RECOVERY TURN (bounded' in m['content']
               for m in messages)


def capture_events(engine):
    events = []
    orig = engine.telemetry.record

    def spy(kind, initiator="model", **fields):
        event = orig(kind, initiator=initiator, **fields)
        events.append(event)
        return event

    engine.telemetry.record = spy
    return events


def find_victim(engine):
    for uid in engine.store.order:
        rec = engine.store.get(uid)
        if getattr(rec, 'archived', False):
            continue
        envelope = getattr(rec, 'envelope', {}) or {}
        if envelope.get('role') == 'assistant' and envelope.get('tool_calls'):
            return uid
    return None


def test_host_usage_bridge_records_true_prompt_size(tmp_path):
    engine, source = make_engine(tmp_path)
    assert engine.last_prompt_tokens == 0
    engine.on_turn_complete(source, usage=reported(30000))
    assert engine.last_prompt_tokens == 30000


def test_working_crossing_arms_and_render_carries_directive_and_tail_echo(tmp_path):
    engine, source = make_engine(tmp_path)
    cross(engine, source)
    messages = engine.select_context(source, conversation_messages=source)
    assert engine._recovery_turn.state == 'offered'
    assert engine._recovery_turn.emitted is True
    assert directive_in(messages)
    # Inline reclaimable spans ride the directive (no clm_status hop).
    assert any('reclaimable spans measured right now' in m.get('content', '')
               for m in messages if isinstance(m.get('content'), str))
    # The tail echo is the final envelope while the turn is offered.
    assert messages[-1]['role'] == 'system'
    assert 'tail reminder' in messages[-1]['content']


def test_wire_prompt_alone_does_not_arm(tmp_path):
    """A huge host wire prompt with a small working set must NOT arm.

    The wire prompt includes fixed overhead (system prompt + tool schemas)
    that clm_compact can never settle; arming on it inflates every real
    session (regression guard for the 2026-10-13 unit-mismatch defect).
    """
    engine, source = make_engine(tmp_path)
    engine.on_turn_complete(source, usage=reported(60000))
    messages = engine.select_context(source, conversation_messages=source)
    assert engine._recovery_turn.state == 'idle'
    assert not directive_in(messages)


def test_wire_prompt_recorded_in_arm_telemetry(tmp_path):
    engine, source = make_engine(tmp_path)
    cross(engine, source)
    engine.on_turn_complete(source, usage=reported(60000))
    events = capture_events(engine)
    engine.select_context(source, conversation_messages=source)
    armed_events = [e for e in events if e.get('event') == 'recovery_turn']
    assert len(armed_events) == 1
    event = armed_events[0]
    assert event['wire_prompt_tokens'] == 60000
    # The crossing measure itself is engine currency, not the wire number.
    assert event['active_tokens'] == event['store_tokens']
    assert event['active_tokens'] > event['usable_tokens']


def test_model_reduction_settles_as_model_and_blocks_rearm_within_episode(tmp_path):
    engine, source = make_engine(tmp_path)
    cross(engine, source)
    engine.select_context(source, conversation_messages=source)
    assert engine._recovery_turn.state == 'offered'
    # Model acts inside the turn: archive the oldest reclaimable group.
    victim = find_victim(engine)
    assert victim
    result = json.loads(engine.handle_tool_call('clm_archive', {'message_ids': [victim],
                                                                'reason': 'recovery'}))
    assert result.get('ok'), result
    # Turn boundary: the offered turn settles as model progress.
    engine.on_turn_complete(source)
    outcome = engine._recovery_turn.outcome
    assert engine._recovery_turn.state == 'idle'
    assert outcome and outcome['selection'] == 'model' and outcome['progress'] is True
    assert outcome['fallback'] is None
    # Same pressure episode: a new crossing does not re-arm (no chained turns).
    engine.on_turn_complete(source, usage=reported(31000))
    messages = engine.select_context(source, conversation_messages=source)
    assert engine._recovery_turn.state == 'idle'
    assert not directive_in(messages)


def test_pressure_clear_retires_episode_and_a_later_crossing_arms_again(tmp_path):
    engine, source = make_engine(tmp_path)
    cross(engine, source)
    engine.select_context(source, conversation_messages=source)
    assert engine._recovery_turn.state == 'offered'
    # Pressure clears because the working set itself drops below usable
    # (model-side archives, not a wire-size report).
    guard = 0
    while engine.store.working_set_tokens() > int(engine.budget.usable_tokens):
        victim = find_victim(engine)
        assert victim, 'ran out of reclaimable groups before clearing pressure'
        result = json.loads(engine.handle_tool_call('clm_archive', {'message_ids': [victim],
                                                                    'reason': 'recovery'}))
        assert result.get('ok'), result
        guard += 1
        assert guard < 100
    engine.on_turn_complete(source)
    outcome = engine._recovery_turn.outcome
    # The clearing archives are model-class ops: settle is model-owned.
    assert outcome and outcome['selection'] == 'model' and outcome['progress'] is True
    # Pressure cleared: the next preparation retires the episode.
    engine.select_context(source, conversation_messages=source)
    assert engine._recovery_turn.used_epoch is None
    assert engine._recovery_turn.outcome is None
    # A later crossing opens exactly one fresh bounded turn.
    cross(engine, source)
    messages = engine.select_context(source, conversation_messages=source)
    assert engine._recovery_turn.state == 'offered'
    assert engine._recovery_turn.episode == 2
    assert directive_in(messages)


def test_mid_turn_usage_calibration_and_satisfied_tail_echo_do_not_loop(tmp_path):
    """During a multi-step recovery turn on the production host, update_from_response
    runs after each API call with wire prompt_tokens (including system + tool schemas),
    and select_context re-runs for the next tool-loop step before on_turn_complete.
    Ensure wire usage does not inflate counter._ratio mid-turn and the tail reminder
    clears once model compaction drops working_set_tokens below usable_tokens.
    """
    engine, source = make_engine(tmp_path)
    cross(engine, source)
    messages = engine.select_context(source, conversation_messages=source)
    assert engine._recovery_turn.state == "offered"
    assert directive_in(messages)
    assert messages[-1]["role"] == "system" and "tail reminder" in messages[-1]["content"]

    # Simulate host turn_usage calling update_from_response after API call #1
    # with 55,000 wire prompt tokens (~30k system + tool schemas).
    engine.update_from_response(reported(55000))
    assert engine.counter._ratio is None

    # Model compacts/archives enough groups inside the turn to clear usable_tokens.
    while engine.store.working_set_tokens() > int(engine.budget.usable_tokens):
        victim = find_victim(engine)
        assert victim
        res = json.loads(engine.handle_tool_call("clm_archive", {"message_ids": [victim], "reason": "recovery"}))
        assert res["ok"], res

    # Next API call in the same turn (before on_turn_complete): directive and tail
    # reminder must not re-fire now that working_set_tokens <= usable_tokens, while
    # the turn remains offered so on_turn_complete settles it as model progress.
    followup = engine.select_context(source, conversation_messages=source)
    assert engine._recovery_turn.state == "offered"
    assert not directive_in(followup)
    assert not (followup[-1]["role"] == "system" and "tail reminder" in followup[-1].get("content", ""))

    engine.on_turn_complete(source, usage=reported(35000))
    outcome = engine._recovery_turn.outcome
    assert engine._recovery_turn.state == "idle"
    assert outcome and outcome["selection"] == "model" and outcome["progress"] is True and outcome["recovered"] is True


def test_render_diagnostic_tracks_tail_divergence_across_episode_calls(tmp_path):
    """H1 instrumentation (2026-10-14): prompt hash + first-divergence offset
    show episode renders diverging only at the tail (directive echo appended),
    not at a top-region notice mutation."""
    engine, source = make_engine(tmp_path)
    # T-1: normal render before the crossing.
    engine.select_context(source, conversation_messages=source)
    diag_a = dict(engine._last_render_diagnostic)
    assert diag_a['prompt_hash'] and diag_a['wire_bytes'] > 0
    assert diag_a['recovery_state'] == 'idle'
    # Cross usable, then the recovery turn's Call 1 (armed, frozen snapshot).
    cross(engine, source)
    engine.select_context(source, conversation_messages=source)
    assert engine._recovery_turn.state == 'offered'
    diag_b = dict(engine._last_render_diagnostic)
    assert diag_b['prompt_hash'] and diag_b['prompt_hash'] != diag_a['prompt_hash']
    assert diag_b['recovery_state'] == 'offered'
    assert diag_b['first_divergence_offset'] is not None
    # Divergence sits at the tail: everything up to the previous render's last
    # message is unchanged; new spans + the directive echo follow it.
    assert diag_b['first_divergence_msg_index'] >= diag_a['message_count'] - 1
    # Mid-turn: a new tool exchange appends; Call 2 must diverge only at the
    # tail (frozen prefix + appended messages + frozen echo).
    cid = 'mid-turn-1'
    source.extend([
        {'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': cid, 'type': 'function',
             'function': {'name': 'read_file', 'arguments': '{}'}}]},
        {'role': 'tool', 'tool_call_id': cid, 'content': 'M' * 400},
    ])
    engine.select_context(source, conversation_messages=source)
    assert engine._recovery_turn.state == 'offered'
    diag_c = dict(engine._last_render_diagnostic)
    assert diag_c['prompt_hash'] != diag_b['prompt_hash']
    assert diag_c['first_divergence_msg_index'] >= diag_b['message_count'] - 1
    assert engine._render_diagnostics[-1]['prompt_hash'] == diag_c['prompt_hash']


def test_mid_turn_archive_defers_physical_rewrite_to_next_turn(tmp_path):
    """H2: model compaction inside the offered turn does not rewrite the
    physical wire mid-turn; the frozen presentation keeps the victim until
    the turn boundary, and T+1 renders without it."""
    engine, source = make_engine(tmp_path)
    cross(engine, source)
    engine.select_context(source, conversation_messages=source)
    assert engine._recovery_turn.state == 'offered'
    victim = find_victim(engine)
    assert victim
    cid = (engine.store.get(victim).envelope.get('tool_calls') or [{}])[0].get('id')
    assert cid
    result = json.loads(engine.handle_tool_call(
        'clm_archive', {'message_ids': [victim], 'reason': 'recovery'}))
    assert result.get('ok'), result
    assert not engine.store.get(victim).active
    # Same turn, next call: frozen presentation still carries the victim.
    call_mid = engine.select_context(source, conversation_messages=source)
    assert cid in json.dumps(call_mid)
    # Turn boundary settles as model progress; T+1 renders without the victim.
    engine.on_turn_complete(source)
    assert engine._recovery_turn.state == 'idle'
    assert engine._recovery_turn.frozen_prefix is None
    call_next = engine.select_context(source, conversation_messages=source)
    assert cid not in json.dumps(call_next)


def test_frozen_tail_echo_byte_identical_across_non_reducing_calls(tmp_path):
    """H1: the directive and tail echo are episode-frozen — a non-reducing
    mid-turn call renders the byte-identical echo (no candidate churn)."""
    engine, source = make_engine(tmp_path)
    cross(engine, source)
    call1 = engine.select_context(source, conversation_messages=source)
    assert engine._recovery_turn.state == 'offered'
    assert call1[-1]['role'] == 'system' and 'tail reminder' in call1[-1]['content']
    echo1 = call1[-1]['content']
    directive1 = engine._recovery_turn.frozen_directive
    assert directive1
    # Non-reducing mid-turn call: a status hop reclaims nothing.
    engine.handle_tool_call('clm_status', {})
    assert engine.store.working_set_tokens() > int(engine.budget.usable_tokens)
    call2 = engine.select_context(source, conversation_messages=source)
    assert engine._recovery_turn.state == 'offered'
    assert call2[-1]['role'] == 'system' and 'tail reminder' in call2[-1]['content']
    assert call2[-1]['content'] == echo1
    assert engine._recovery_turn.frozen_directive == directive1


def test_scaffolding_retirement_and_resilient_uids_single_compact(tmp_path):
    """H3: one clm_compact with a hallucinated uid succeeds (valid-subset
    filtering), compacts the victim, and retires CLM scaffolding + orphaned
    acks in the same call."""
    engine, source = make_engine(tmp_path)
    # Inject CLM scaffolding + a short orphan ack before crossing.
    cid = 'scaffold-1'
    source.extend([
        {'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': cid, 'type': 'function',
             'function': {'name': 'clm_status', 'arguments': '{}'}}]},
        {'role': 'tool', 'tool_call_id': cid, 'content': 'scaffold payload ' * 20},
        {'role': 'assistant', 'content': 'ack'},
    ])
    engine.on_turn_complete(source)
    cross(engine, source)
    engine.select_context(source, conversation_messages=source)
    assert engine._recovery_turn.state == 'offered'

    def uid_of(pred):
        for uid in engine.store.order:
            rec = engine.store.get(uid)
            env = getattr(rec, 'envelope', {}) or {}
            if pred(rec, env):
                return uid
        return None

    scaffold_uid = uid_of(lambda rec, env: any(
        (tc.get('function') or {}).get('name') == 'clm_status'
        for tc in (env.get('tool_calls') or []) if isinstance(tc, dict)))
    ack_uid = uid_of(lambda rec, env: env.get('role') == 'assistant'
                     and not env.get('tool_calls') and env.get('content') == 'ack')
    assert scaffold_uid and ack_uid
    victim = find_victim(engine)
    assert victim
    members = engine.store.resolve_groups([victim])
    group_tokens = sum(int(engine.store.get(u).token_count or 0) for u in members)
    before = engine.store.working_set_tokens()
    result = json.loads(engine.handle_tool_call(
        'clm_compact', {'message_ids': ['bogus-uid-xyz', victim],
                        'summary': 'span summary with key=value facts preserved'}))
    assert result.get('ok'), result
    # Resilient UID filtering: the valid victim resolved; the typo dropped.
    resolved = result.get('message_ids') or []
    assert victim in resolved
    assert 'bogus-uid-xyz' not in resolved
    # Scaffolding + orphan ack retired by the same single call.
    assert not engine.store.get(scaffold_uid).active
    assert not engine.store.get(ack_uid).active
    # Combined reduction is at least the victim group's own tokens.
    after = engine.store.working_set_tokens()
    assert before - after >= group_tokens
