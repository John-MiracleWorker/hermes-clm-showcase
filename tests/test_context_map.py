"""Map provenance survives real archival, rewind and persisted-state boundaries."""
from copy import deepcopy
import json

import pytest

from hermes_clm import CLMEngine
from hermes_clm.config import CLMConfig
from hermes_clm.policies import PolicyViolation
from hermes_clm.protocol import RecoveryRequired
from hermes_clm.store import ContextStore
from conftest import _make_config


def originals():
    return [
        {'role': 'system', 'content': 'preserve the originals'},
        {'role': 'user', 'content': 'root task'},
        {'role': 'assistant', 'content': None, 'reasoning': 'retained reasoning',
         'tool_calls': [{'id': 'read-1', 'type': 'function',
                         'function': {'name': 'read_file', 'arguments': ' { "path": "report" } '}}]},
        {'role': 'tool', 'tool_call_id': 'read-1', 'name': 'read_file',
         'content': 'RAW-DETAIL-7391', 'opaque': {'sequence': [3, 7]}},
        {'role': 'user', 'content': 'continue the task'},
        {'role': 'assistant', 'content': 'ongoing work'},
    ]


def engine_for(tmp_path, *, enabled=True, conversation='map-conversation'):
    # Existing config mutation is intentional: RED reaches real old operations,
    # instead of failing construction because a new keyword is absent.
    cfg = _make_config(tmp_path, protect_latest_user_turns=0,
                       protect_latest_assistant_turns=0)
    cfg.context_map_enabled = enabled
    eng = CLMEngine(cfg, str(tmp_path))
    eng.on_session_start('map-session', conversation_id=conversation)
    eng.store.ingest_messages(originals())
    return eng


def index_for(store):
    assert 'context_map' in store.state_dict(), 'Real source operations must retain their authoritative map'
    assert hasattr(store, 'context_map_index'), 'The independently validated shared index is required'
    return store.context_map_index


def group_uids(store):
    return [store._by_index[2], store._by_index[3]]


def reference_for(index, uids):
    refs = index.to_dict()['archive_refs']
    matches = [ref_id for ref_id, ref in refs.items() if ref['source_uids'] == uids]
    assert len(matches) == 1, 'A complete raw source group has one stable host reference'
    return matches[0]


def test_archive_result_covers_whole_original_group(tmp_path):
    eng = engine_for(tmp_path)
    uids = group_uids(eng.store)
    eng.apply_archive([uids[1]])
    index = index_for(eng.store)
    ref = reference_for(index, uids)
    audit = index.validate_refs([ref], require_current=True)
    assert audit['refs'][ref]['source_uids'] == uids
    nodes = index.to_dict()['nodes']
    assert any(ref in node['archive_refs'] for node in nodes.values())
    assert [eng.store.get(u).envelope for u in uids] == originals()[2:4]


def test_compaction_map_binds_raw_and_summary_sources(tmp_path):
    eng = engine_for(tmp_path)
    uids = group_uids(eng.store)
    result = eng.apply_compact([uids[1]], 'A terse result summary')
    index = index_for(eng.store)
    data = index.to_dict()
    ref = reference_for(index, uids)
    assert any(ref in node['archive_refs'] and result['summary_id'] in node['summary_refs']
               for node in data['nodes'].values())
    assert eng.store.summary(result['summary_id']).derivation_source_uids == uids
    assert 'RAW-DETAIL-7391' not in json.dumps(data)
    assert 'A terse result summary' not in json.dumps(data)


def test_body_compaction_and_preserved_history_have_source_coverage(tmp_path):
    eng = engine_for(tmp_path)
    uids = group_uids(eng.store)
    eng.apply_compact([uids[1]], 'body summary one', form='body')
    eng.apply_compact([uids[1]], 'body summary two', form='body')
    index = index_for(eng.store)
    ref = reference_for(index, uids)
    assert any(ref in node['archive_refs'] for node in index.to_dict()['nodes'].values())
    assert eng.store.get(uids[1]).body_compaction_history
    assert eng.store.get(uids[1]).envelope == originals()[3]
    index.validate_refs()


def test_host_reference_survives_restore_archive_and_append(tmp_path):
    eng = engine_for(tmp_path)
    uids = group_uids(eng.store)
    eng.apply_archive([uids[1]])
    ref = reference_for(index_for(eng.store), uids)
    eng.apply_restore([uids[1]])
    assert reference_for(index_for(eng.store), uids) == ref
    eng.apply_archive([uids[1]])
    eng.store.ingest_messages(originals() + [{'role': 'user', 'content': 'new turn'}])
    assert reference_for(index_for(eng.store), uids) == ref
    assert index_for(eng.store).validate_refs([ref], require_current=True)['refs'][ref]['source_uids'] == uids


def test_active_tool_group_retains_proven_branch_witness_after_rewind(tmp_path):
    eng = engine_for(tmp_path)
    old = group_uids(eng.store)
    changed = originals()
    changed[2]['tool_calls'][0]['function']['arguments'] = '{"path":"corrected-report"}'
    eng.store.ingest_messages(changed)
    index = index_for(eng.store)
    ref = reference_for(index, old)
    assert old[0] not in eng.store.order and old[1] in eng.store.order
    assert index.validate_refs([ref])['refs'][ref]['source_uids'] == old
    with pytest.raises(PolicyViolation) as caught:
        index.validate_refs([ref], require_current=True)
    assert caught.value.code == 'historic_branch_conflict'
    assert eng.store.get(old[0]).envelope == originals()[2]


def test_legacy_current_backfill_is_deterministic_and_preserves_originals(tmp_path):
    old = engine_for(tmp_path / 'old', enabled=False)
    uids = group_uids(old.store)
    old.apply_archive([uids[1]])
    state = old.store.state_dict()
    assert 'context_map' not in state
    cfg = deepcopy(old.config)
    cfg.context_map_enabled = True
    a, b = ContextStore(cfg), ContextStore(cfg)
    a.load_state_dict(state)
    b.load_state_dict(state)
    ia, ib = index_for(a), index_for(b)
    assert reference_for(ia, uids) == reference_for(ib, uids)
    assert ia.to_dict()['nodes'].keys() == ib.to_dict()['nodes'].keys()
    assert [a.get(u).envelope for u in uids] == originals()[2:4]
    ia.validate_refs()


def test_legacy_unproved_historic_tool_join_requires_recovery(tmp_path):
    old = engine_for(tmp_path / 'old', enabled=False)
    changed = originals()
    changed[2]['tool_calls'][0]['function']['arguments'] = '{"path":"other"}'
    old.store.ingest_messages(changed)
    state = old.store.state_dict()
    cfg = deepcopy(old.config)
    cfg.context_map_enabled = True
    loaded = ContextStore(cfg)
    loaded.load_state_dict(state)
    assert loaded.recovery_required == 'context_map_provenance_unverifiable'
    assert loaded.legacy_state == state
    with pytest.raises(RecoveryRequired):
        loaded.require_envelopes()


def test_checkpoint_restart_preserves_refs_with_feature_disabled(tmp_path):
    eng = engine_for(tmp_path)
    uids = group_uids(eng.store)
    eng.apply_archive([uids[1]])
    index = index_for(eng.store)
    ref = reference_for(index, uids)
    eng.store.save(eng.config.state_root)
    cfg = deepcopy(eng.config)
    cfg.context_map_enabled = False
    loaded = ContextStore.load(cfg.state_root, eng.store.conversation_id, cfg)
    assert not loaded.recovery_required
    replay = index_for(loaded)
    assert reference_for(replay, uids) == ref
    assert replay.to_dict() == index.to_dict()
    loaded.mark_archived([loaded._by_index[5]])
    replay.validate_refs()
    assert reference_for(replay, [loaded._by_index[5]])


def test_path_reuse_allocates_host_ids_and_returns_detached_nodes(tmp_path):
    eng = engine_for(tmp_path)
    eng.apply_archive([group_uids(eng.store)[1]])
    index = index_for(eng.store)
    first = index.create_node(['Reports', 'Q4'], group_uids(eng.store),
                              description='quarterly reports', keywords=['finance'])
    second = index.create_node(['ＲＥＰＯＲＴＳ', 'q4'], group_uids(eng.store))
    assert second.node_id == first.node_id
    data = first.to_dict()
    data['title'] = 'caller changed label'
    assert index.get_node(first.node_id).title == 'Q4'
    root = index.get_node(index.to_dict()['root_id'])
    assert root.parent_id is None
    parent = index.get_node(first.parent_id)
    assert parent.child_ids.count(first.node_id) == 1
    foreign = engine_for(tmp_path / 'foreign', conversation='different-conversation')
    foreign.apply_archive([group_uids(foreign.store)[1]])
    assert first.node_id not in index_for(foreign.store).to_dict()['nodes']


@pytest.mark.parametrize('path', [[], [''] , ['topic'] * 6, ['x' * 97]])
def test_invalid_path_cannot_leave_partially_allocated_nodes(tmp_path, path):
    eng = engine_for(tmp_path)
    eng.apply_archive([group_uids(eng.store)[1]])
    index = index_for(eng.store)
    before = eng.store.state_dict()
    with pytest.raises((ValueError, PolicyViolation)):
        index.create_node(path, group_uids(eng.store))
    assert eng.store.state_dict() == before


@pytest.mark.parametrize('sources', ['partial', 'empty', 'unknown'])
def test_public_create_rejects_unverified_source_leaf(tmp_path, sources):
    eng = engine_for(tmp_path)
    uids = group_uids(eng.store)
    eng.apply_archive([uids[1]])
    index = index_for(eng.store)
    requested = {'partial': [uids[1]], 'empty': [], 'unknown': ['not-a-source']}[sources]
    before = eng.store.state_dict()
    with pytest.raises((ValueError, PolicyViolation)):
        index.create_node(['unproved'], requested)
    assert eng.store.state_dict() == before


def test_attach_and_update_validate_without_tool_dispatch(tmp_path):
    eng = engine_for(tmp_path)
    eng.apply_archive([group_uids(eng.store)[1]])
    index = index_for(eng.store)
    ref = reference_for(index, group_uids(eng.store))
    node = index.create_node(['explicit'], group_uids(eng.store))
    index.attach_archive_ref(node.node_id, ref)
    assert index.get_node(node.node_id).archive_refs.count(ref) == 1
    index.update_node(node.node_id, description='known description', keywords=['evidence'])
    assert index.get_node(node.node_id).description == 'known description'
    before = eng.store.state_dict()
    with pytest.raises((ValueError, PolicyViolation)):
        index.attach_archive_ref(node.node_id, 'foreign-ref')
    with pytest.raises((ValueError, PolicyViolation)):
        index.update_node(node.node_id, description='x' * 257)
    assert eng.store.state_dict() == before


@pytest.mark.parametrize('corruption', ['cycle', 'dangling_child', 'partial_group', 'scope', 'orphan'])
def test_corrupt_map_import_is_atomic(tmp_path, corruption):
    eng = engine_for(tmp_path)
    eng.apply_archive([group_uids(eng.store)[1]])
    index = index_for(eng.store)
    data = index.to_dict()
    root = data['root_id']
    leaf = next(n for n in data['nodes'].values() if n['archive_refs'])
    if corruption == 'cycle':
        data['nodes'][root]['parent_id'] = leaf['node_id']
        leaf['child_ids'].append(root)
    elif corruption == 'dangling_child':
        data['nodes'][root]['child_ids'].append('missing-node')
    elif corruption == 'partial_group':
        ref = reference_for(index, group_uids(eng.store))
        data['archive_refs'][ref]['source_uids'] = [group_uids(eng.store)[1]]
    elif corruption == 'scope':
        data['scope']['conversation_id'] = 'different-conversation'
    else:
        leaf['archive_refs'] = []
        leaf['summary_refs'] = []
    before = eng.store.state_dict()
    with pytest.raises((ValueError, PolicyViolation)):
        type(index).from_dict(data, store=eng.store)
    assert eng.store.state_dict() == before


def test_corrupt_checkpoint_does_not_replace_accepted_state(tmp_path):
    eng = engine_for(tmp_path)
    eng.apply_archive([group_uids(eng.store)[1]])
    index_for(eng.store)
    accepted = eng.store.state_dict()
    corrupt = deepcopy(accepted)
    corrupt['context_map']['archive_refs'].clear()
    with pytest.raises((ValueError, PolicyViolation)):
        eng.store.load_state_dict(corrupt)
    assert eng.store.state_dict() == accepted


def test_direct_map_writer_retains_exact_admission_without_counter(tmp_path):
    eng = engine_for(tmp_path)
    eng.apply_archive([group_uids(eng.store)[1]])
    index = index_for(eng.store)
    eng.config.budget_mode = 'exact'
    eng.store._budget_validator = None
    before = eng.store.state_dict()
    with pytest.raises(PolicyViolation) as caught:
        index.create_node(['refused'], group_uids(eng.store))
    assert caught.value.code == 'exact_counter_unconfigured'
    assert eng.store.state_dict() == before


@pytest.mark.parametrize('operation', ['create', 'attach', 'update', 'get',
                                      'search', 'render', 'validate', 'export', 'import'])
def test_every_map_api_rejects_invalid_canonical_source_independently(tmp_path, operation):
    eng = engine_for(tmp_path)
    uids = group_uids(eng.store)
    eng.apply_archive([uids[1]])
    index = index_for(eng.store)
    data = index.to_dict()
    ref = reference_for(index, uids)
    node = next(n['node_id'] for n in data['nodes'].values() if ref in n['archive_refs'])
    tampered = deepcopy(originals()[3])
    tampered['content'] = 'changed after canonical admission'
    eng.store.get(uids[1]).set_envelope(tampered)
    before = eng.store.state_dict()
    calls = {
        'create': lambda: index.create_node(['rejected'], uids),
        'attach': lambda: index.attach_archive_ref(node, ref),
        'update': lambda: index.update_node(node, title='rejected'),
        'get': lambda: index.get_node(node),
        'search': lambda: index.search_nodes('archive', max_results=5),
        'render': lambda: index.render_snapshot(),
        'validate': lambda: index.validate_refs(),
        'export': lambda: index.to_dict(),
        'import': lambda: type(index).from_dict(data, store=eng.store),
    }
    with pytest.raises(ValueError, match='invalid_canonical_identity'):
        calls[operation]()
    assert eng.store.state_dict() == before


@pytest.mark.parametrize('error', [RuntimeError, KeyboardInterrupt, SystemExit])
def test_direct_map_mutation_rolls_back_after_budget_validator_exit(tmp_path, error):
    eng = engine_for(tmp_path)
    eng.apply_archive([group_uids(eng.store)[1]])
    index = index_for(eng.store)
    eng.config.budget_mode = 'exact'
    before = eng.store.state_dict()
    def admission(store, operation, phase):
        if phase == 'after':
            assert any(n['title'] == 'new-label' for n in index.to_dict()['nodes'].values())
            raise error('after actual map allocation')
    eng.store._budget_validator = admission
    with pytest.raises(error):
        index.create_node(['new-label'], group_uids(eng.store))
    assert eng.store.state_dict() == before


def test_default_off_preserves_existing_archive_behavior(tmp_path):
    eng = engine_for(tmp_path, enabled=False)
    uids = group_uids(eng.store)
    eng.apply_archive([uids[1]])
    assert 'context_map' not in eng.store.state_dict()
    assert all(eng.store.get(u).archived for u in uids)
    assert [eng.store.get(u).envelope for u in uids] == originals()[2:4]


@pytest.mark.parametrize('field,bad', [
    ('context_map_enabled', 1), ('context_map_enabled', 'yes'),
    ('context_map_token_cap', True), ('context_map_token_cap', 2001),
    ('context_map_token_target', 0), ('context_map_token_target', 2001),
    ('context_map_max_depth', 6), ('context_map_max_depth', 0),
    ('context_map_description_limit', 257), ('context_map_description_limit', -1),
    ('context_map_max_results', 6), ('context_map_max_results', 0),
    ('context_map_mode', 'embeddings'), ('context_map_browse', 'yes'),
])
def test_config_rejects_unbounded_or_wrongly_typed_map_options(field, bad):
    cfg = CLMConfig()
    setattr(cfg, field, bad)
    with pytest.raises(ValueError):
        cfg._validate()


def test_actual_hermes_config_section_enables_map_consumer(tmp_path):
    cfg = CLMConfig.from_hermes({'clm': {'context_map_enabled': True,
         'protect_latest_user_turns': 0, 'protect_latest_assistant_turns': 0}}, str(tmp_path))
    eng = CLMEngine(cfg, str(tmp_path))
    eng.on_session_start('map-session', conversation_id='configured-map')
    eng.store.ingest_messages(originals())
    eng.apply_archive([group_uids(eng.store)[1]])
    index_for(eng.store).validate_refs()
