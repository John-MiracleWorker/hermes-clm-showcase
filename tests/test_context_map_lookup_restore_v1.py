"""Tests for Task 4.4: Deterministic lookup, optional browse policy, and reference restoration."""
from copy import deepcopy
import json
from unittest.mock import patch
import pytest

from hermes_clm.config import CLMConfig
from hermes_clm.policies import PolicyViolation
from test_context_map import engine_for, originals, group_uids, reference_for
from test_archive_lookup import call as lookup_call, dataset, node, ranked_fixture, paginated_fixture
from test_context_map_restore import restore
from test_context_map_piggyback_slice2 import whole_snapshot


def test_lookup_deterministic_weights_and_tie_breaks(tmp_path):
    """Spec 4.4: Local matching uses frozen weights (title=8, keywords=6, path=4, desc=2) and (-score, node_id) tie breaks."""
    e, uids = dataset(tmp_path, count=10)
    # Node 1: title match only -> 8 points
    n1 = node(e, ['Target Alpha'], [uids[0]])
    # Node 2: keywords match only -> 6 points
    n2 = node(e, ['Topic Beta'], [uids[1]], keywords=['target'])
    # Node 3: path match only -> 4 points
    n3 = node(e, ['Target Parent', 'Child Gamma'], [uids[2]])
    # Node 4: description match only -> 2 points
    n4 = node(e, ['Topic Delta'], [uids[3]], description='detailed target finding')

    res = lookup_call(e, 'target')
    assert res['ok'] is True
    assert res['normalized_query'] == 'target'
    assert len(res['results']) == 4

    # Let's inspect components:
    c1 = res['results'][0]['score_components']
    assert c1['title'] == 8
    assert c1['keywords'] == 0
    assert c1['path'] == 0
    assert c1['description'] == 0

    c2 = res['results'][1]['score_components']
    assert c2['keywords'] == 6

    c3 = res['results'][2]['score_components']
    assert c3['path'] == 4

    c4 = res['results'][3]['score_components']
    assert c4['description'] == 2

    # Tie break check with identical score
    # Create two nodes with identical title score
    n_tie_b = node(e, ['Identical Zeta'], [uids[4]])
    n_tie_a = node(e, ['Identical Alpha'], [uids[5]])
    res_tie = lookup_call(e, 'identical')
    assert res_tie['ok'] is True
    assert len(res_tie['results']) == 2
    # Alphabetical order of node_id breaks ties
    assert res_tie['results'][0]['host']['node_id'] < res_tie['results'][1]['host']['node_id']


def test_lookup_recency_and_access_bonuses(tmp_path):
    """Spec 4.4: Recency bonus (+2 newest, +1 2nd newest) and access bonus (min(access_count, 3))."""
    e, uids = dataset(tmp_path, count=10)

    # Create 3 nodes with distinct timestamps:
    with patch('hermes_clm.retrieval.context_map.utc_now_iso', return_value='2026-02-01T12:00:00+00:00'):
        n1 = e.store.context_map_index.create_node(['ScoreTest One'], [uids[0]])
    with patch('hermes_clm.retrieval.context_map.utc_now_iso', return_value='2026-01-15T12:00:00+00:00'):
        n2 = e.store.context_map_index.create_node(['ScoreTest Two'], [uids[1]])
    with patch('hermes_clm.retrieval.context_map.utc_now_iso', return_value='2026-01-01T12:00:00+00:00'):
        n3 = e.store.context_map_index.create_node(['ScoreTest Three'], [uids[2]])

    res1 = lookup_call(e, 'ScoreTest')
    assert res1['ok'] is True
    assert len(res1['results']) == 3

    node_map = {r['host']['node_id']: r for r in res1['results']}
    assert node_map[n1.node_id]['score_components']['recency'] == 2
    assert node_map[n2.node_id]['score_components']['recency'] == 1
    assert node_map[n3.node_id]['score_components']['recency'] == 0

    # Before lookup 1, access_count was 0 for all nodes
    assert node_map[n1.node_id]['score_components']['access'] == 0
    assert node_map[n2.node_id]['score_components']['access'] == 0
    assert node_map[n3.node_id]['score_components']['access'] == 0

    # Lookup 1 presented n1, n2, n3, so their access_count incremented to 1
    res2 = lookup_call(e, 'ScoreTest')
    node_map2 = {r['host']['node_id']: r for r in res2['results']}
    assert node_map2[n1.node_id]['score_components']['access'] == 1

    # Lookup 2 incremented access_count to 2
    res3 = lookup_call(e, 'ScoreTest')
    node_map3 = {r['host']['node_id']: r for r in res3['results']}
    assert node_map3[n1.node_id]['score_components']['access'] == 2

    # Lookup 3 incremented access_count to 3
    res4 = lookup_call(e, 'ScoreTest')
    node_map4 = {r['host']['node_id']: r for r in res4['results']}
    assert node_map4[n1.node_id]['score_components']['access'] == 3

    # Lookup 4 increments access_count to 4, but bonus is capped at 3 (min(count, 3))
    res5 = lookup_call(e, 'ScoreTest')
    node_map5 = {r['host']['node_id']: r for r in res5['results']}
    assert node_map5[n1.node_id]['score_components']['access'] == 3


def test_lookup_result_cap_and_cursor_pagination(tmp_path):
    """Spec 4.4: Result cap is 5; cursor pagination returns deterministic pages without duplicates."""
    e, nodes = paginated_fixture(tmp_path)
    # Default max_results is 5
    page1 = lookup_call(e, 'needle')
    assert page1['ok'] is True
    assert len(page1['results']) == 5
    assert page1['max_results'] == 5
    assert page1['next_cursor'] is not None

    # Page 2
    page2 = lookup_call(e, 'needle', cursor=page1['next_cursor'])
    assert page2['ok'] is True
    assert len(page2['results']) > 0

    # Ensure no duplicate archive_refs across pages
    refs_p1 = {r['host']['archive_ref'] for r in page1['results']}
    refs_p2 = {r['host']['archive_ref'] for r in page2['results']}
    assert refs_p1.isdisjoint(refs_p2)

    # Reject invalid max_results: >5, 0, negative, non-int
    res_overflow = lookup_call(e, 'needle', max_results=6)
    assert res_overflow['ok'] is False
    assert res_overflow['error'] == 'invalid_context_map_result_limit'

    res_zero = lookup_call(e, 'needle', max_results=0)
    assert res_zero['ok'] is False
    assert res_zero['error'] == 'invalid_context_map_result_limit'

    res_str = lookup_call(e, 'needle', max_results='5')
    assert res_str['ok'] is False
    assert res_str['error'] == 'invalid_context_map_result_limit'


def test_lookup_rejects_stale_and_corrupt_cursors(tmp_path):
    """Spec 4.4: Stale or corrupt cursors reject cleanly without modifying store."""
    e, nodes = paginated_fixture(tmp_path)
    page1 = lookup_call(e, 'needle', max_results=2)
    cursor = page1['next_cursor']
    assert cursor is not None
    before = whole_snapshot(e)

    # 1. Tampered / corrupt cursor
    corrupt = cursor[:-4] + 'zzzz'
    res_corrupt = lookup_call(e, 'needle', cursor=corrupt)
    assert res_corrupt['ok'] is False
    assert res_corrupt['error'] in ('invalid_lookup_cursor', 'stale_lookup_cursor')
    assert whole_snapshot(e) == before

    # 2. Query mismatch -> stale_lookup_cursor
    res_mismatch = lookup_call(e, 'otherquery', cursor=cursor)
    assert res_mismatch['ok'] is False
    assert res_mismatch['error'] == 'stale_lookup_cursor'
    assert whole_snapshot(e) == before


def test_lookup_does_not_reveal_answer_bodies_or_restore(tmp_path):
    """Spec 4.4: Lookup returns only metadata and token estimates; does not restore or leak secret text."""
    e = engine_for(tmp_path)
    source = originals()
    secret = 'SUPER-SECRET-PASSCODE-9999'
    source[3]['content'] = secret
    e.store.ingest_messages(source)
    uids = group_uids(e.store)

    e.apply_archive([uids[1]], map_path=['SecurityVault', 'Keys'], map_description='Public metadata only')

    # Message must be archived and NOT active
    assert e.store.get(uids[1]).archived is True

    res = lookup_call(e, 'Keys')
    assert res['ok'] is True
    assert len(res['results']) == 1

    # Secret content must NOT be in the lookup reply
    serialized = json.dumps(res)
    assert secret not in serialized

    # Lookup must NOT have restored the message
    assert e.store.get(uids[1]).archived is True
    assert e.store.get(uids[1]).active is False


def test_browse_map_omitted_unless_verified_navigation_support():
    """Spec 4.4: clm_browse_map is omitted unless verified navigation support exists."""
    cfg = CLMConfig()
    assert cfg.context_map_browse is False

    # Enabling context_map_browse without verified navigation support must be rejected by config
    cfg.context_map_browse = True
    with pytest.raises(ValueError) as exc:
        cfg.validate_context_map()
    assert 'context_map_browse requires separately verified navigation support' in str(exc.value)


def test_restore_accepts_validated_archive_refs_and_closes_tool_groups(tmp_path):
    """Spec 4.4: clm_restore accepts validated archive_refs, restores complete tool groups, and records provenance."""
    e = engine_for(tmp_path)
    e.config.emit_nudges = False
    uids = group_uids(e.store)

    e.apply_archive([uids[1]], map_path=['ToolGroup', 'Run'], keywords=['cli'])
    ref = reference_for(e.store.context_map_index, uids)

    # Perform lookup to generate lookup receipt
    lookup_res = lookup_call(e, 'Run')
    assert lookup_res['ok'] is True

    # Restore via archive_ref
    restore_res = restore(e, [ref], reason='Retrieve original execution output')
    assert restore_res['ok'] is True
    assert restore_res['archive_refs'] == [ref]
    assert restore_res['restored_ids'] == uids
    # All messages in the tool group must be restored together
    for uid in uids:
        msg = e.store.get(uid)
        assert msg.active is True
        assert msg.archived is False


def test_restore_rejects_unknown_and_historic_refs_atomically(tmp_path):
    """Spec 4.4: Unknown refs reject with unknown_archive_ref; historic refs reject with historic_branch_conflict."""
    e = engine_for(tmp_path)
    e.config.emit_nudges = False
    uids = group_uids(e.store)
    e.apply_archive([uids[1]], map_path=['Historical', 'Record'])
    ref = reference_for(e.store.context_map_index, uids)
    before = whole_snapshot(e)

    # 1. Unknown reference
    res_unk = restore(e, ['non_existent_archive_ref'])
    assert res_unk['ok'] is False
    assert res_unk['error'] == 'unknown_archive_ref'
    assert whole_snapshot(e) == before

    # 2. Historic branch conflict
    new_source = originals()
    new_source[3]['content'] = 'BRANCH DIVERGED CONTENT'
    e.store.ingest_messages(new_source)
    before_diverged = whole_snapshot(e)

    res_hist = restore(e, [ref])
    assert res_hist['ok'] is False
    assert res_hist['error'] == 'historic_branch_conflict'
    assert whole_snapshot(e) == before_diverged


def test_restore_budget_exceeded_rejects_atomically(tmp_path):
    """Spec 4.4: Exact restore requirements report headroom and reject atomically when budget is exceeded."""
    e = engine_for(tmp_path)
    e.config.emit_nudges = False
    uids = group_uids(e.store)
    e.apply_archive([uids[1]], map_path=['LargeData', 'Chunk'])
    ref = reference_for(e.store.context_map_index, uids)
    before = whole_snapshot(e)

    # 1. Budget controller overflow -> budget_exceeded with headroom and tokens_required reported
    e.config.max_restoration_tokens = 100000
    e.budget.active_target = e.store.working_set_tokens() + 1
    before = whole_snapshot(e)
    res_budget = restore(e, [ref], reason='Attempt overflow restore')
    assert res_budget['ok'] is False
    assert res_budget['error'] == 'budget_exceeded'
    assert res_budget['tokens_required'] > 1
    assert 'headroom_tokens' in res_budget
    assert 'hint' in res_budget
    assert whole_snapshot(e) == before
    for uid in uids:
        assert e.store.get(uid).archived is True

    # 2. Artificially set max_restoration_tokens very low -> max_restoration_exceeded
    e.budget.active_target = 100000
    e.config.max_restoration_tokens = 1
    before_max = whole_snapshot(e)
    res_max = restore(e, [ref], reason='Attempt oversize restore')
    assert res_max['ok'] is False
    assert res_max['error'] == 'max_restoration_exceeded'
    assert whole_snapshot(e) == before_max
    for uid in uids:
        assert e.store.get(uid).archived is True
