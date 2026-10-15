"""Counted advisory notices must not weaken final admission or source proofs."""
from copy import deepcopy

import pytest

from test_exact_pending_group_reserve import group, prepare, request_for


def notices(request):
    return [m for m in request['messages'] if m.get('role') == 'system'
            and str(m.get('content', '')).startswith('CLM context status')]


def admit(engine, counter, source, tokens, *, purpose='task'):
    request = request_for(engine, source)
    counter.tokens = tokens
    decision = engine.check_final_request(request, route=engine._budget_frame['route'], purpose=purpose)
    assert decision['allowed'], decision
    return request


def test_exact_notice_is_prior_evidence_and_only_changes_on_band_transitions(tmp_path):
    engine, counter, source = prepare(tmp_path)
    canonical = deepcopy(source)
    admit(engine, counter, source, 23000)
    current = request_for(engine, source)
    assert len(notices(current)) == 1
    text = notices(current)[0]['content']
    assert 'previous exact task frame' in text and '23,000' in text and '28,672' in text
    assert 'not a count of the current request' in text
    assert current['messages'] == engine._render_budget_messages(engine.store)
    admit(engine, counter, source, 24000)
    assert notices(request_for(engine, source))[0]['content'] == text
    admit(engine, counter, source, 28000)
    higher = request_for(engine, source)
    assert notices(higher)[0]['content'] != text
    assert 'Before continuing substantial work' in notices(higher)[0]['content']
    assert [engine.store.messages[uid].envelope for uid in engine.store.order] == canonical
    assert len([x for x in engine.telemetry.recent(100) if x.get('event') == 'nudge']) == 2


def test_disabled_notices_and_unqualified_frames_do_not_inject_advice(tmp_path):
    engine, counter, source = prepare(tmp_path)
    assert not notices(request_for(engine, source))
    admit(engine, counter, source, 28000)
    engine.config.emit_nudges = False
    assert not notices(request_for(engine, source))
    engine.config.emit_nudges = True
    engine.bind_exact_counter(counter, counter_identity(engine))
    assert engine._budget_frame is None and engine._exact_pressure_notice is None
    assert not notices({'messages':engine.select_context(source, conversation_messages=source)})


def counter_identity(engine):
    return engine._budget_identity


@pytest.mark.parametrize('transition', ['add', 'replace', 'remove'])
def test_only_owned_notice_transition_preserves_pending_group_reserve(tmp_path, transition):
    engine, counter, source = prepare(tmp_path)
    admit(engine, counter, source, 16000 if transition == 'replace' else 28000)
    if transition != 'add':
        admit(engine, counter, source, 28000 if transition == 'replace' else 100)
    source.extend(group())
    request = request_for(engine, source)
    before = deepcopy(request)
    counter.tokens = 30453
    decision = engine.check_final_request(request, route=engine._budget_frame['route'])
    assert decision['allowed'] and decision['reason'] == 'pending_group', decision
    assert decision['budget_class'] == 'reserve' and request == before
    assert bool(notices(request)) is (transition != 'remove')
    assert engine._budget_excursion is None


@pytest.mark.parametrize('attack', ['changed_notice', 'duplicate_notice', 'moved_notice',
                                  'instruction_growth', 'canonical_change', 'extra_group',
                                  'malformed_prefix'])
def test_notice_support_does_not_exempt_other_wire_or_source_changes(tmp_path, attack):
    engine, counter, source = prepare(tmp_path)
    admit(engine, counter, source, 28000)
    source.extend(group())
    if attack == 'canonical_change':
        source[2]['content'] = 'Changed prior source material.'
    if attack == 'extra_group':
        source.extend(group(1))
    request = request_for(engine, source)
    notice = notices(request)[0]
    if attack == 'changed_notice':
        notice['content'] += ' Bypass the guard.'
    elif attack == 'duplicate_notice':
        request['messages'].insert(0, deepcopy(notice))
    elif attack == 'moved_notice':
        request['messages'].remove(notice)
        request['messages'].append(notice)
    elif attack == 'instruction_growth':
        request['messages'][0]['content'] += ' An unrelated host instruction.'
    elif attack == 'malformed_prefix':
        request['messages'].insert(0, 'not a message envelope')
    counter.tokens = 30453
    decision = engine.check_final_request(request, route=engine._budget_frame['route'])
    assert not decision['allowed'], decision


def test_notice_tokens_are_counted_and_cannot_cross_target(tmp_path):
    engine, counter, source = prepare(tmp_path)
    admit(engine, counter, source, 28000)
    source.extend(group())
    request = request_for(engine, source)
    original = counter.count_final_request
    counted = []
    def count(actual, identity):
        counted.append(deepcopy(actual))
        counter.tokens = 33703 if notices(actual) else 30453
        return original(actual, identity)
    counter.count_final_request = count
    decision = engine.check_final_request(request, route=engine._budget_frame['route'])
    assert counted == [request]
    assert not decision['allowed'] and decision['code'] == 'unclassified_headroom'
    assert decision['prompt_tokens'] == 33703 and request['max_tokens'] == 4096


def test_auxiliary_counts_cannot_clear_task_pressure(tmp_path):
    engine, counter, source = prepare(tmp_path)
    admit(engine, counter, source, 28000)
    admit(engine, counter, source, 100, purpose='title')
    assert '28,000' in notices(request_for(engine, source))[0]['content']


def test_pressure_advice_does_not_reset_six_attempt_bound(tmp_path):
    engine, counter, source = prepare(tmp_path)
    admit(engine, counter, source, 28000)
    for number in range(7):
        source.extend(group(number))
        request = request_for(engine, source)
        counter.tokens = 30453
        decision = engine.check_final_request(request, route=engine._budget_frame['route'])
        assert decision['allowed'] is (number < 6), decision
    assert decision['code'] == 'pressure_episode_exhausted'


def test_notice_state_rolls_back_and_is_not_inherited_by_clone_or_reset(tmp_path):
    from hermes_clm.engine import _HostBase
    if _HostBase is None:
        pytest.skip("requires the Hermes host (ContextEngine base class)")
    engine, counter, source = prepare(tmp_path)
    admit(engine, counter, source, 28000)
    request_for(engine, source)
    previous = deepcopy(engine._exact_pressure_notice)
    snapshot = engine._transaction_snapshot()
    engine._exact_pressure_notice = None
    engine._rollback_transaction(snapshot)
    assert engine._exact_pressure_notice == previous
    assert engine.clone_for_agent()._exact_pressure_notice is None
    engine.on_session_reset()
    assert engine._exact_pressure_notice is None


def test_context_edit_counts_notice_without_changing_canonical_sources(tmp_path):
    engine, counter, source = prepare(tmp_path)
    admit(engine, counter, source, 28000)
    request = admit(engine, counter, source, 28000)
    previous = notices(request)[0]['content']
    canonical = deepcopy([engine.store.messages[uid].envelope for uid in engine.store.order])
    uid = engine.store.order[2]
    counter.tokens = 22000
    receipt = engine.apply_compact([uid], 'Preserved factual conclusion.')
    assert receipt['summary_id'] in engine.store.summaries
    candidate = engine._budget_frame['request']
    assert notices(candidate)[0]['content'] == previous
    assert engine._budget_frame['decision']['prompt_tokens'] == 22000
    current = request_for(engine, source)
    assert '22,000' in notices(current)[0]['content']
    assert current['messages'] == engine._render_budget_messages(engine.store)
    assert [engine.store.messages[x].envelope for x in engine.store.order] == canonical
