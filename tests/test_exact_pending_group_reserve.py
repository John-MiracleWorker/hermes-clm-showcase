"""Regression for V14's 30,453-token tool-result completion refusal.

The fixed counter tests admission semantics only; it is not provider parity.
"""
from copy import deepcopy

import pytest

from test_exact_atomic_v6 import setup


def group(number=0):
    ids = [f'read-{number}-a', f'read-{number}-b']
    return [{'role': 'assistant', 'content': '', 'tool_calls': [
        {'id': cid, 'type': 'function', 'function': {'name': 'read_file', 'arguments': '{}'}}
        for cid in ids]}, *[{'role': 'tool', 'tool_call_id': cid,
                            'content': 'A complete source result.'} for cid in ids]]


def prepare(tmp_path):
    engine, counter, _ = setup(tmp_path)
    source = [deepcopy(engine.store.messages[uid].envelope) for uid in engine.store.order]
    return engine, counter, source


def request_for(engine, source):
    request = deepcopy(engine._budget_frame['request'])
    request['messages'] = engine.select_context(source, conversation_messages=source)
    return request


def test_completed_ordinary_tool_group_may_use_internal_reserve(tmp_path):
    engine, counter, source = prepare(tmp_path)
    source.extend(group())
    request = request_for(engine, source)
    before = deepcopy(request)
    counter.tokens = 30453
    decision = engine.check_final_request(request, route=engine._budget_frame['route'])
    assert decision['allowed'], decision
    assert decision['budget_class'] == 'reserve' and decision['reason'] == 'pending_group'
    assert decision['prompt_tokens'] == 30453 and request == before
    assert engine._pressure_episode.attempts == 1
    assert engine._budget_excursion is None, 'Derived completion must not become a standing grant'


@pytest.mark.parametrize('change', ['no_growth', 'user_tail', 'two_groups', 'rewritten_prefix',
                                  'wire_growth', 'schema_growth', 'auxiliary', 'above_target'])
def test_pending_completion_does_not_license_unrelated_growth(tmp_path, change):
    engine, counter, source = prepare(tmp_path)
    if change != 'no_growth':
        source.extend(group())
    if change == 'user_tail':
        source.append({'role': 'user', 'content': 'Another task'})
    if change == 'two_groups':
        source.extend(group(1))
    if change == 'rewritten_prefix':
        source[2]['content'] = 'Changed earlier source'
    request = request_for(engine, source)
    if change == 'wire_growth':
        request['messages'][0]['content'] += ' Unrelated instruction growth'
    if change == 'schema_growth':
        request['tools'] = [{'unqualified': 'new schema'}]
    counter.tokens = 32769 if change == 'above_target' else 30453
    decision = engine.check_final_request(request, route=engine._budget_frame['route'],
                                         purpose='title' if change == 'auxiliary' else 'task')
    assert not decision['allowed'], decision


def test_reserve_cannot_be_reused_without_another_completed_group(tmp_path):
    engine, counter, source = prepare(tmp_path)
    source.extend(group())
    request = request_for(engine, source)
    counter.tokens = 30453
    route = engine._budget_frame['route']
    assert engine.check_final_request(request, route=route)['allowed']
    assert engine.check_final_request(request, route=route)['code'] == 'reserve_reason_required'


def test_pending_group_keeps_pressure_episode_bounded(tmp_path):
    engine, counter, source = prepare(tmp_path)
    counter.tokens = 30453
    for i in range(7):
        source.extend(group(i))
        request = request_for(engine, source)
        decision = engine.check_final_request(request, route=engine._budget_frame['route'])
        assert decision['allowed'] is (i < 6), decision
    assert decision['code'] == 'pressure_episode_exhausted'


def test_incomplete_parallel_result_is_never_sent(tmp_path):
    engine, counter, source = prepare(tmp_path)
    source.extend(group())
    request = request_for(engine, source)
    request['messages'].pop()
    counter.tokens = 30453
    decision = engine.check_final_request(request, route=engine._budget_frame['route'])
    assert not decision['allowed'] and decision['code'] == 'invalid_final_protocol'
