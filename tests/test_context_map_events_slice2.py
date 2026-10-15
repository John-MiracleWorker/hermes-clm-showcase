"""Actual source-edit events and their physical publication are transactional."""
from contextlib import contextmanager
from copy import deepcopy
import builtins
import json
from pathlib import Path

import pytest

from hermes_clm.policies import PolicyViolation
from hermes_clm.store import ContextStore
from test_context_map import engine_for, group_uids, originals
from test_context_map_piggyback_slice2 import invoke, whole_snapshot


OPERATIONS = ('archive', 'summary', 'body')
KINDS = {'archive': 'archive', 'summary': 'compact', 'body': 'compact_body'}


def events(eng):
    path = Path(eng._state_root) / eng.store.conversation_id / 'events.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def assert_delta(event, before, after, sources):
    assert 'context_map_delta' in event, 'The real event must bind the committed Context Map change'
    delta = event['context_map_delta']
    assert delta['revision_before'] == before['revision']
    assert delta['revision_after'] == after['revision']
    assert delta['source_uids'] == sources
    assert delta['node_ids'] == [uid for uid, node in after['nodes'].items() if node != before['nodes'].get(uid)]
    assert delta['archive_refs'] == [ref for ref, record in after['archive_refs'].items()
                                     if set(record['source_uids']).issubset(sources)]
    introduced = set(after['presentations']) - set(before['presentations'])
    for uid, node in after['nodes'].items():
        introduced.update(set(node['presentation_refs']) - set(before['nodes'].get(uid, {}).get('presentation_refs', [])))
    assert delta['presentation_refs'] == [pid for pid in after['presentations'] if pid in introduced]


@pytest.mark.parametrize('operation', OPERATIONS)
@pytest.mark.parametrize('semantic', [False, True])
def test_model_dispatch_event_binds_actual_map_delta_and_initiator(tmp_path, operation, semantic):
    eng = engine_for(tmp_path)
    before = eng.store.context_map_index.to_dict()
    metadata = {'map_path': ['Project', 'Result']} if semantic else {}
    # Caller-owned attribution is not part of the tool surface.
    metadata['initiator'] = 'host-emergency'
    assert invoke(eng, operation, metadata)['ok'] is True
    event = events(eng)[-1]
    assert event['event'] == KINDS[operation] and event['initiator'] == 'model'
    assert_delta(event, before, eng.store.context_map_index.to_dict(), group_uids(eng.store))


@pytest.mark.parametrize('operation', OPERATIONS)
def test_disabled_authoring_retains_delta_for_existing_map(tmp_path, operation):
    eng = engine_for(tmp_path)
    eng.config.context_map_enabled = False
    before = eng.store.context_map_index.to_dict()
    assert invoke(eng, operation)['ok'] is True
    assert_delta(events(eng)[-1], before, eng.store.context_map_index.to_dict(), group_uids(eng.store))


def pressured_engine(tmp_path):
    eng = engine_for(tmp_path)
    source = originals()
    source[3]['content'] = 'large retained source ' * 1000
    source.extend([{'role': 'user', 'content': 'newest user'}, {'role': 'assistant', 'content': 'newest answer'}])
    eng.store.ingest_messages(source)
    eng.budget.active_target = 1000
    assert eng.store.working_set_tokens() > eng.budget.hard_limit
    return eng, source


@pytest.mark.parametrize('route', ['manual', 'pressure'])
def test_real_host_compression_routes_keep_actual_attribution(tmp_path, route):
    eng, source = pressured_engine(tmp_path)
    before = eng.store.context_map_index.to_dict()
    if route == 'manual':
        eng.compress(source)
    else:
        eng._overflow = True
        eng._overflow_retries = eng.config.overflow_compaction_retries + 1
        eng.select_context(deepcopy(source))
    actual = [event for event in events(eng) if event['event'] == 'emergency_compaction']
    assert len(actual) == 1 and actual[0]['source_message_ids']
    event = actual[0]
    assert event['initiator'] == ('host-manual' if route == 'manual' else 'host-emergency')
    assert_delta(event, before, eng.store.context_map_index.to_dict(), event['source_message_ids'])


@pytest.mark.parametrize('operation', OPERATIONS)
def test_no_map_event_preserves_existing_contract_control(tmp_path, operation):
    eng = engine_for(tmp_path, enabled=False)
    assert invoke(eng, operation)['ok'] is True
    assert 'context_map_delta' not in events(eng)[-1]


def publication_setup(tmp_path, *, absent=False, persistence=True, logging=True):
    eng = engine_for(tmp_path)
    eng._persist()
    base = Path(eng._state_root) / eng.store.conversation_id
    if absent:
        for name in ('state.json', 'events.jsonl'):
            (base / name).unlink(missing_ok=True)
        base.rmdir()
    else:
        (base / 'state.json').chmod(0o640)
        (base / 'events.jsonl').chmod(0o640)
    sentinel = Path(eng._state_root) / 'unrelated-sentinel'
    sentinel.write_bytes(b'preserve unrelated file')
    sentinel.chmod(0o640)
    eng.config.persistence = persistence
    eng.config.event_log = logging
    return eng, base, sentinel


def intercept_physical_append(monkeypatch, eng, base, failure):
    original = builtins.open
    reached = []
    state = base / 'state.json'
    old_state = state.read_bytes() if state.exists() else None

    @contextmanager
    def failing_open(file, mode='r', *args, **kwargs):
        with original(file, mode, *args, **kwargs) as stream:
            class PartialWriter:
                def write(self, text):
                    if eng.config.persistence:
                        assert state.read_bytes() != old_state, 'Failure must follow the real state replacement'
                    else:
                        assert (state.read_bytes() if state.exists() else None) == old_state
                    written = stream.write(text[:max(1, len(text) // 2)])
                    stream.flush()
                    assert written > 0
                    reached.append({'partial_event_written': True, 'state_saved': eng.config.persistence})
                    raise failure('after physical partial event append')
            yield PartialWriter()

    def open_hook(file, mode='r', *args, **kwargs):
        if str(file) == str(base / 'events.jsonl') and mode == 'a':
            return failing_open(file, mode, *args, **kwargs)
        return original(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, 'open', open_hook)
    return reached, original


@pytest.mark.parametrize('operation', OPERATIONS)
@pytest.mark.parametrize('failure', [OSError, KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize('absent', [False, True])
def test_physical_event_failure_restores_all_source_map_runtime_and_owned_files(tmp_path, monkeypatch, operation, failure, absent):
    eng, base, sentinel = publication_setup(tmp_path, absent=absent)
    before = whole_snapshot(eng)
    reached, _ = intercept_physical_append(monkeypatch, eng, base, failure)
    caught = None
    try:
        invoke(eng, operation, {'map_path': ['Project', 'Physical']}, direct=True)
    except BaseException as exc:
        caught = exc
    assert reached, 'The injected physical write must actually execute'
    assert type(caught) is failure, 'A mapped source edit must not accept a failed physical event write'
    assert whole_snapshot(eng) == before
    assert sentinel.read_bytes() == b'preserve unrelated file' and sentinel.stat().st_mode & 0o777 == 0o640


@pytest.mark.parametrize('operation', OPERATIONS)
def test_event_failure_is_strict_without_checkpoint_persistence(tmp_path, monkeypatch, operation):
    eng, base, _ = publication_setup(tmp_path, persistence=False)
    before = whole_snapshot(eng)
    reached, _ = intercept_physical_append(monkeypatch, eng, base, OSError)
    caught = None
    try:
        invoke(eng, operation, direct=True)
    except BaseException as exc:
        caught = exc
    assert reached and type(caught) is OSError
    assert whole_snapshot(eng) == before


@pytest.mark.parametrize('operation', OPERATIONS)
def test_logging_disabled_source_edit_has_no_physical_event_control(tmp_path, monkeypatch, operation):
    eng, base, _ = publication_setup(tmp_path, logging=False)
    prior_events = (base / 'events.jsonl').read_bytes()
    reached, _ = intercept_physical_append(monkeypatch, eng, base, OSError)
    assert invoke(eng, operation, {'map_path': ['Without', 'Events']})['ok'] is True
    assert not reached and (base / 'events.jsonl').read_bytes() == prior_events
    saved = json.loads((base / 'state.json').read_text())
    assert saved['context_map'] == eng.store.context_map_index.to_dict()


@pytest.mark.parametrize('operation', OPERATIONS)
def test_failed_file_rollback_retains_existing_owner_until_successful_retry(tmp_path, monkeypatch, operation):
    eng, base, _ = publication_setup(tmp_path)
    before = whole_snapshot(eng)
    reached, original_open = intercept_physical_append(monkeypatch, eng, base, OSError)
    restore = eng.store._restore_session_files
    attempts = []
    def once(files):
        attempts.append(True)
        if len(attempts) == 1:
            raise OSError('injected owned file rollback failure')
        return restore(files)
    monkeypatch.setattr(ContextStore, '_restore_session_files', staticmethod(once))
    caught = None
    try:
        invoke(eng, operation, {'map_path': ['Rollback', 'Owner']}, direct=True)
    except BaseException as exc:
        caught = exc
    assert reached and isinstance(caught, OSError) and attempts
    assert eng._pending_direct_rollback is not None
    with pytest.raises(PolicyViolation) as blocked:
        invoke(eng, operation, direct=True)
    assert blocked.value.code == 'context_rollback_incomplete'
    monkeypatch.setattr(builtins, 'open', original_open)
    eng.retry_context_rollback()
    assert eng._pending_direct_rollback is None
    assert whole_snapshot(eng) == before
    assert invoke(eng, operation, {'map_path': ['Rollback', 'Retry']})['ok'] is True
    assert len([e for e in events(eng) if e['event'] == KINDS[operation]]) == 1
