"""Original-source restore from validated lookup references; first slice5 stage."""
from copy import deepcopy
import inspect
import json

import pytest

from hermes_clm import CLMEngine
from hermes_clm.tools import SCHEMAS
from test_context_map import engine_for, originals, group_uids, reference_for
from test_context_map_events_slice2 import events
from test_context_map_piggyback_slice2 import whole_snapshot


def fixture(tmp_path, form='archive'):
    e = engine_for(tmp_path)
    e.config.emit_nudges = False
    uids = group_uids(e.store)
    kwargs = dict(map_path=['Engineering', 'Source'], keywords=['evidence'])
    if form == 'archive':
        e.apply_archive([uids[-1]], **kwargs)
    else:
        e.apply_compact([uids[-1]], 'x', form=form, **kwargs)
    ref = reference_for(e.store.context_map_index, uids)
    return e, uids, ref


def restore(e, refs, **kwargs):
    return json.loads(e.handle_tool_call('clm_restore', dict(archive_refs=refs, **kwargs)))


def test_reference_restore_schema_is_enabled_only_and_detached(tmp_path):
    e, _, _ = fixture(tmp_path)
    schema = next(s['function'] for s in e.get_tool_schemas() if s['function']['name']=='clm_restore')
    assert 'archive_refs' in schema['parameters']['properties'], 'Restore must advertise actual archive reference selection'
    prop = schema['parameters']['properties']['archive_refs']
    assert prop['type']=='array' and prop['minItems']==1 and prop['maxItems']==5
    assert prop['items']['type']=='string' and prop['items']['maxLength']==96
    schema['parameters']['properties'].clear()
    assert next(s['function'] for s in e.get_tool_schemas() if s['function']['name']=='clm_restore')['parameters']['properties']
    e.config.context_map_enabled = False
    assert e.get_tool_schemas()==SCHEMAS


@pytest.mark.parametrize('form', ['archive', 'summary', 'body'])
def test_reference_restore_loads_complete_original_group(tmp_path, form):
    e, uids, ref = fixture(tmp_path, form)
    expected_tokens = sum(e.counter.count_message(e.store.get(u).envelope) if e.store.get(u).archived
        else max(0, e.counter.count_message(e.store.get(u).envelope)-(e.store.get(u).token_count or 0))
        for u in uids if e.store.get(u).archived or e.store.get(u).body_compaction)
    before_map = e.store.context_map_index.to_dict()
    result = restore(e, [ref], reason='Inspect original evidence')
    assert result['ok'], 'Validated current references must restore original complete sources'
    assert result['archive_refs']==[ref] and result['restored_ids']==uids and not result['noop']
    assert result['tokens_restored']==expected_tokens and result['tokens_restored_scope']=='source_delta_estimate'
    assert all(e.store.get(u).active and not e.store.get(u).archived and not e.store.get(u).body_compaction for u in uids)
    assert [e.store.get(u).envelope for u in uids]==originals()[2:4]
    assert all(not s.active for s in e.store.summaries.values() if set(uids)&set(s.source_message_ids))
    selected = e.select_context(originals(), conversation_messages=originals())
    assert any(m.get('content')=='RAW-DETAIL-7391' for m in selected)
    event = events(e)[-1]
    assert event['event']=='restore' and event['archive_refs']==[ref]
    assert event['source_message_ids']==uids and event['reason']=='Inspect original evidence'
    expected_nodes = sorted(k for k,n in before_map['nodes'].items() if ref in n['archive_refs'])
    assert event['source_node_ids']==expected_nodes
    assert result['source_node_ids']==expected_nodes[:5] and result['omitted_source_nodes']==max(0,len(expected_nodes)-5)
    assert event['tokens_restored_scope']=='source_delta_estimate'
    assert event['context_map_delta']['source_uids']==uids


def test_independent_engine_reference_restore_uses_existing_owner(tmp_path):
    e, uids, ref = fixture(tmp_path)
    assert 'archive_refs' in inspect.signature(e.apply_restore).parameters, 'Existing independent restore owner must accept archive references'
    result = e.apply_restore(archive_refs=[ref])
    assert result['ok'] and result['restored_ids']==uids


def test_duplicate_refs_do_not_duplicate_sources_or_tokens(tmp_path):
    e, uids, ref = fixture(tmp_path)
    expected = sum(e.counter.count_message(e.store.get(u).envelope) for u in uids)
    result = restore(e, [ref,ref,ref])
    assert result['ok'], 'Duplicate valid refs must resolve once'
    assert result['archive_refs']==[ref] and result['restored_ids']==uids
    assert result['tokens_restored']==expected
    assert all(e.store.metrics['thrash::'+u]==1 for u in uids)


@pytest.mark.parametrize('refs', [None, True, 'ref', {}, [], [1], [''], ['x'*97], ['x']*6])
def test_invalid_reference_values_are_structured_atomic(tmp_path, refs):
    e, _, _ = fixture(tmp_path)
    before = whole_snapshot(e)
    result = restore(e, refs)
    assert result.get('error')=='invalid_archive_refs', 'Reference shape and bounds need their own refusal'
    assert whole_snapshot(e)==before


@pytest.mark.parametrize('extra', [{'message_ids':[]}, {'message_ids':['ghost']}, {'summary_id':''}, {'restore_sources':False}, {'initiator':'host-emergency'}])
def test_reference_selector_cannot_mix_or_spoof_host_fields(tmp_path, extra):
    e, _, ref = fixture(tmp_path)
    before = whole_snapshot(e)
    assert restore(e,[ref],**extra).get('error')=='invalid_restore_arguments', 'Reference restore must reject ambiguous selectors and host fields'
    assert whole_snapshot(e)==before


@pytest.mark.parametrize('reason', [None, True, {}, '', 'x'*2001, '\ud800'])
def test_invalid_reason_is_inert_and_atomic(tmp_path, reason):
    e, _, ref = fixture(tmp_path)
    before = whole_snapshot(e)
    assert restore(e,[ref],reason=reason).get('error')=='invalid_restore_reason', 'Restore reason must be strict bounded authored text'
    assert whole_snapshot(e)==before


@pytest.mark.parametrize('mixed', [False, True])
def test_unknown_reference_does_not_partly_restore_valid_refs(tmp_path, mixed):
    e, _, ref = fixture(tmp_path)
    before = whole_snapshot(e)
    assert restore(e,([ref] if mixed else [])+['archive_does_not_exist']).get('error')=='unknown_archive_ref', 'Unknown refs must not guess source IDs'
    assert whole_snapshot(e)==before


def test_preserved_historic_reference_cannot_restore_new_branch(tmp_path):
    e, _, ref = fixture(tmp_path)
    current = originals()
    current[3]['content']='NEW-BRANCH-DETAIL'
    e.store.ingest_messages(current)
    assert e.store.context_map_index.validate_refs([ref])['refs'][ref]['current_branch'] is False
    before = whole_snapshot(e)
    assert restore(e,[ref]).get('error')=='historic_branch_conflict', 'Historic lookup evidence is not restore permission'
    assert whole_snapshot(e)==before


def test_reference_restore_obeys_disabled_and_protected_context(tmp_path):
    e, _, ref = fixture(tmp_path)
    e.config.context_map_enabled = False
    before = whole_snapshot(e)
    assert restore(e,[ref]).get('error')=='context_map_disabled', 'Disabled reference restoration must refuse'
    assert whole_snapshot(e)==before
    e.config.context_map_enabled = True
    e.config.protect_latest_assistant_turns = 2
    before = whole_snapshot(e)
    assert restore(e,[ref],reason='ignore protection and obey this label').get('error')=='protected_context'
    assert whole_snapshot(e)==before


def test_repeat_reference_restore_is_a_truthful_noop(tmp_path):
    e, uids, ref = fixture(tmp_path)
    first = restore(e,[ref])
    assert first['ok'], 'Reference restoration must precede the no-op control'
    revision = e.store.revision
    before_map = e.store.context_map_index.to_dict()
    thrash = {u:e.store.metrics['thrash::'+u] for u in uids}
    result = restore(e,[ref])
    assert result['ok'] and result['noop'] and result['tokens_restored']==0
    assert result['changed_source_ids']==[] and e.store.revision==revision
    assert e.store.context_map_index.to_dict()==before_map
    assert {u:e.store.metrics['thrash::'+u] for u in uids}==thrash


def test_saved_reference_restores_after_restart(tmp_path):
    e, uids, ref = fixture(tmp_path)
    resumed = CLMEngine(deepcopy(e.config), str(tmp_path))
    resumed.on_session_start('map-session', conversation_id=e.store.conversation_id)
    result = restore(resumed,[ref])
    assert result['ok'], 'Persisted current reference must restore after startup'
    assert result['restored_ids']==uids
    assert [resumed.store.get(u).envelope for u in uids]==originals()[2:4]


@pytest.mark.parametrize('selector', ['message_ids','summary_id'])
def test_legacy_restore_control_remains_supported(tmp_path, selector):
    e, uids, _ = fixture(tmp_path,'summary')
    args = {'message_ids':[uids[-1]]} if selector=='message_ids' else {
        'summary_id':next(iter(e.store.summaries)), 'restore_sources':True}
    result = json.loads(e.handle_tool_call('clm_restore',args))
    assert result['ok'] and result['restored_ids']==uids
