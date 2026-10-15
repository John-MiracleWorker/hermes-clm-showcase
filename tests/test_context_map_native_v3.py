"""Real native projection, coordinate lineage and persisted map controls.

These are component lifecycle checks, not provider sends or full Hermes-loop
qualification. Descriptors declare known source occurrences independently.
"""
import os
from copy import deepcopy
import hashlib
import json
import sys

import pytest

from hermes_clm import CLMEngine
from hermes_clm.protocol import envelope_uid
from hermes_clm.store import ContextStore
from conftest import _make_config
from test_context_map import originals, group_uids, index_for, reference_for


@pytest.fixture(autouse=True)
def actual_hermes_dependencies(monkeypatch):
    # The guarded runner permits this existing SDK directory read-only. Use real
    # packages, with candidate Hermes/CLM still before it on the import path.
    monkeypatch.setattr(sys, 'path', sys.path + [
        os.path.expanduser('~/.hermes/hermes-agent/venv/lib/python3.11/site-packages')])
    try:  # these cases exercise the paired Hermes runtime's native projection
        from agent.turn_context import _reissue_narrative_projection  # noqa: F401
    except Exception:
        pytest.skip("paired Hermes runtime not present (native projection support missing)")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def projection(source):
    sources = [{'index': i, 'uid': envelope_uid(i, message),
                'exact_digest': digest(message)} for i, message in enumerate(source)]
    rows = [{'request_index': i, 'wire_digest': digest(message),
             'origin': {'kind': 'source', 'source': {'index': i, 'uid': sources[i]['uid']},
                        'adaptations': []},
             'policy': {'essential': False, 'archive': True,
                        'body': message['role'] == 'tool', 'narrative': True}}
            for i, message in enumerate(source)]
    return {'version': 1, 'mode': 'native', 'conversation_digest': digest(source),
            'request_digest': digest(source), 'incoming_index': 1, 'sources': sources,
            'rows': rows, 'omissions': [], 'narrative_slots': []}


def native_engine(tmp_path, *, enabled=True):
    cfg = _make_config(tmp_path, protect_latest_user_turns=0,
                       protect_latest_assistant_turns=0, context_map_enabled=enabled)
    eng = CLMEngine(cfg, str(tmp_path))
    eng.prepare_context_origin('native')
    eng.on_session_start('native-session', conversation_id='native-map')
    source = originals()
    selected = eng.select_context_with_projection(deepcopy(source),
        conversation_messages=source, incoming_message=source[1],
        budget_tokens=0, source_projection=projection(source))
    from hermes_clm.prompts import CLM_SYSTEM_GUIDANCE
    guidance = {'role': 'system', 'content': CLM_SYSTEM_GUIDANCE}
    assert selected.count(guidance) == 1
    assert [m for m in selected if m != guidance] == source
    assert source == originals()
    return eng


def test_native_map_restart_preserves_coordinate_lineage_and_reference(tmp_path):
    eng = native_engine(tmp_path)
    uids = group_uids(eng.store)
    eng.apply_archive([uids[1]])
    ref = reference_for(index_for(eng.store), uids)
    state = eng.store.state_dict()
    assert state['version'] == 6 and state['origin_mode'] == 'native'
    assert state['coordinate_origin']['mode'] == 'canonical_native'
    eng.on_session_end('native-session', originals())
    resumed = CLMEngine(deepcopy(eng.config), str(tmp_path))
    resumed.prepare_context_origin('native')
    resumed.on_session_start('native-session', conversation_id='native-map')
    assert resumed.store.coordinate_origin == state['coordinate_origin']
    assert index_for(resumed.store).to_dict() == state['context_map']
    assert index_for(resumed.store).validate_refs([ref], require_current=True)['refs'][ref]['source_uids'] == uids
    assert [resumed.store.get(u).envelope for u in uids] == originals()[2:4]


def test_native_version5_backfill_retains_admitted_origin_and_sources(tmp_path):
    old = native_engine(tmp_path, enabled=False)
    uids = group_uids(old.store)
    old.apply_archive([uids[1]])
    state = old.store.state_dict()
    assert state['version'] == 5 and 'context_map' not in state
    old.on_session_end('native-session', originals())
    cfg = deepcopy(old.config)
    cfg.context_map_enabled = True
    resumed = CLMEngine(cfg, str(tmp_path))
    resumed.prepare_context_origin('native')
    resumed.on_session_start('native-session', conversation_id='native-map')
    ref = reference_for(index_for(resumed.store), uids)
    assert resumed.store.state_dict()['version'] == 6
    assert resumed.store.coordinate_origin == state['coordinate_origin']
    assert index_for(resumed.store).validate_refs([ref], require_current=True)['refs'][ref]['source_uids'] == uids
    assert [resumed.store.get(u).envelope for u in uids] == originals()[2:4]


@pytest.mark.parametrize('corruption', ['standalone', 'marker', 'map_scope'])
def test_native_map_origin_corruption_cannot_adopt_candidate(tmp_path, corruption):
    eng = native_engine(tmp_path)
    eng.apply_archive([group_uids(eng.store)[1]])
    before = eng.store.state_dict()
    candidate = deepcopy(before)
    if corruption == 'standalone':
        candidate['origin_mode'] = 'standalone'
        candidate['coordinate_origin'] = None
    elif corruption == 'marker':
        candidate['coordinate_origin']['producer'] = 'invented'
    else:
        candidate['context_map']['scope']['origin_mode'] = 'standalone'
    with pytest.raises(ValueError):
        eng.store.load_state_dict(candidate, context_origin_mode='native')
    assert eng.store.state_dict() == before
    index_for(eng.store).validate_refs()


def test_native_same_conversation_session_transition_retains_map_references(tmp_path):
    eng = native_engine(tmp_path)
    uids = group_uids(eng.store)
    eng.apply_archive([uids[1]])
    ref = reference_for(index_for(eng.store), uids)
    source = originals() + [{'role': 'user', 'content': 'new session follows the same conversation'}]
    successor = CLMEngine(deepcopy(eng.config), str(tmp_path))
    successor.prepare_context_origin('native')
    successor.on_session_start('successor-session', conversation_id='native-map',
        previous_session={'engine': eng, 'session_id': 'native-session', 'messages': source})
    assert successor.store.session_id == 'successor-session'
    assert reference_for(index_for(successor.store), uids) == ref
    assert index_for(successor.store).validate_refs([ref], require_current=True)['refs'][ref]['source_uids'] == uids
    assert successor.store.get(successor.store.order[-1]).envelope == source[-1]
    successor.on_session_end('successor-session', source)
    replay = CLMEngine(deepcopy(eng.config), str(tmp_path))
    replay.prepare_context_origin('native')
    replay.on_session_start('successor-session', conversation_id='native-map')
    assert reference_for(index_for(replay.store), uids) == ref
    assert [replay.store.get(u).envelope for u in uids] == originals()[2:4]
