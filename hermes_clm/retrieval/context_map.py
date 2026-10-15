"""Host-owned topology and retained source-group provenance.

The facade is deliberately not store state. Its JSON data participates in the
same store preimage as source residency, and public writers use the existing
accepted mutation boundary. This index supplies metadata, never source bodies.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime
import hashlib
import json
import unicodedata

from .. import invariants
from ..models import utc_now_iso
from ..protocol import RecoveryRequired, envelope_digest, envelope_uid, validate_json
from ..store import _accepted_mutation

_OMITTED_LIMIT = object()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def _scope(store):
    if not isinstance(store.conversation_id, str) or not store.conversation_id:
        raise ValueError('context_map_scope_unconfigured')
    if not isinstance(store.session_id, str):
        raise ValueError('context_map_scope_unconfigured')
    mode = 'native' if store.coordinate_origin is not None or store._origin_mode == 'native' else 'standalone'
    if store._context_map is not None:
        # IDs retain their creation-session anchor when the host admits another
        # session of this same canonical conversation. Current session_id is
        # lifecycle telemetry; changing it must not rekey retained sources.
        scope = store._context_map.get('scope') if type(store._context_map) is dict else None
        if (type(scope) is not dict or set(scope) != {'conversation_id', 'session_id', 'origin_mode'}
                or scope['conversation_id'] != store.conversation_id
                or scope['origin_mode'] != mode or not isinstance(scope['session_id'], str)):
            raise ValueError('foreign_context_map_scope')
        return scope
    return {'conversation_id': store.conversation_id, 'session_id': store.session_id,
            'origin_mode': mode}


def _norm(value):
    return unicodedata.normalize('NFKC', value).casefold().strip()


def _text(value, limit, *, empty=True):
    if type(value) is not str or len(value) > limit or (not empty and not _norm(value)):
        raise ValueError('invalid_context_map_label')
    try:
        value.encode('utf-8')
    except UnicodeError:
        raise ValueError('invalid_context_map_label') from None
    return value


def _integer(value, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError('invalid_context_map_integer')


def _indices(values):
    if type(values) is not list:
        raise ValueError('invalid_context_map_coordinates')
    for value in values:
        _integer(value)


def _strings(value, *, nonempty=False):
    if type(value) is not list or any(not isinstance(s, str) or not s for s in value):
        raise ValueError('invalid_context_map_ids')
    if len(value) != len(set(value)) or (nonempty and not value):
        raise ValueError('invalid_context_map_ids')


def _timestamp(value, *, nullable=False):
    if value is None and nullable:
        return
    if not isinstance(value, str):
        raise ValueError('invalid_context_map_timestamp')
    try:
        if datetime.fromisoformat(value).tzinfo is None:
            raise ValueError()
    except ValueError:
        raise ValueError('invalid_context_map_timestamp') from None


def _keywords(values):
    if not isinstance(values, (list, tuple)) or len(values) > 16:
        raise ValueError('invalid_context_map_keywords')
    result = [_text(v, 64, empty=False) for v in values]
    if len({_norm(v) for v in result}) != len(result):
        raise ValueError('duplicate_context_map_keyword')
    return result


NODE_FIELDS = frozenset({
    'node_id', 'parent_id', 'title', 'description', 'child_ids', 'archive_refs',
    'summary_refs', 'presentation_refs', 'diagnostic_refs', 'keywords', 'token_estimate',
    'created_at', 'updated_at', 'last_accessed_at', 'access_count',
    'provenance_ids', 'links',
})
MAP_FIELDS = frozenset({'version', 'scope', 'revision', 'node_seq', 'archive_seq',
                        'root_id', 'nodes', 'archive_refs', 'witnesses',
                        'presentations', 'diagnostics', 'recovery_required'})


@dataclass(frozen=True)
class ContextMapNode:
    node_id: str
    parent_id: str | None
    title: str
    description: str
    child_ids: tuple
    archive_refs: tuple
    summary_refs: tuple
    presentation_refs: tuple
    diagnostic_refs: tuple
    keywords: tuple
    token_estimate: int
    created_at: str
    updated_at: str
    last_accessed_at: str | None
    access_count: int
    provenance_ids: tuple
    links: tuple

    @classmethod
    def _from_record(cls, record):
        values = deepcopy(record)
        for key in ('child_ids', 'archive_refs', 'summary_refs', 'presentation_refs',
                    'diagnostic_refs', 'keywords', 'provenance_ids', 'links'):
            values[key] = tuple(values[key])
        return cls(**values)

    def to_dict(self):
        result = deepcopy(self.__dict__)
        for key in ('child_ids', 'archive_refs', 'summary_refs', 'presentation_refs',
                    'diagnostic_refs', 'keywords', 'provenance_ids', 'links'):
            result[key] = list(result[key])
        return result


def _new_node(data, parent_id, title, description='', keywords=()):
    data['node_seq'] += 1
    node_id = f"cm_node_{_digest(data['scope'])[:16]}_{data['node_seq']:08d}"
    stamp = utc_now_iso()
    node = dict(node_id=node_id, parent_id=parent_id, title=title,
                description=description, child_ids=[], archive_refs=[],
                summary_refs=[], presentation_refs=[], diagnostic_refs=[], keywords=list(keywords),
                token_estimate=0, created_at=stamp, updated_at=stamp,
                last_accessed_at=None, access_count=0, provenance_ids=[], links=[])
    data['nodes'][node_id] = node
    if parent_id is not None:
        data['nodes'][parent_id]['child_ids'].append(node_id)
    return node


def _empty_map(store):
    data = {'version': 1, 'scope': _scope(store), 'revision': 0, 'node_seq': 0,
            'archive_seq': 0, 'root_id': '', 'nodes': {}, 'archive_refs': {},
            'witnesses': {}, 'presentations': {}, 'diagnostics': {}, 'recovery_required': None}
    data['root_id'] = _new_node(data, None, 'Context map')['node_id']
    return data


def _ensure(store):
    if store._context_map is None:
        store._context_map = _empty_map(store)
    if store._context_map['scope'] != _scope(store):
        raise ValueError('foreign_context_map_scope')
    return store._context_map


def _witness(store, snapshot, uids, authority, branch_digest):
    invariants.source_groups(snapshot, uids)
    records = snapshot.messages
    payload = {'source_uids': list(uids),
               'envelope_digests': [envelope_digest(records[u]['envelope']) for u in uids],
               'source_indices': [records[u]['index'] for u in uids],
               'scope_digest': _digest(_scope(store)), 'authority': authority,
               'observed_branch_digest': branch_digest}
    return _digest(payload), payload


def _uncertified_sources(snapshot):
    """Only the observed branch can prove whether its own groups completed."""
    validation = invariants.protocol(snapshot)
    invalid = {v.message_index for v in validation.violations
               if v.code != 'interrupted_tool_group'}
    certified = set()
    for group in validation.groups:
        if group.complete and not invalid.intersection(group.source_indices):
            try:
                invariants.source_groups(snapshot, list(group.source_uids))
            except ValueError:
                continue
            certified.update(group.source_uids)
    for uid in snapshot.order:
        record = snapshot.messages[uid]
        if record['role'] != 'tool' and not record['envelope'].get('tool_calls'):
            try:
                invariants.source_groups(snapshot, [uid])
            except ValueError:
                continue
            certified.add(uid)
    return [u for u in snapshot.order if u not in certified]


def _capture_diagnostic_observation(store, data, snapshot):
    sources = _uncertified_sources(snapshot)
    if not sources:
        return
    payload = {'observed_source_uids': list(snapshot.order),
               'envelope_digests': [envelope_digest(snapshot.messages[u]['envelope'])
                                    for u in snapshot.order],
               'uncertified_source_uids': sources,
               'scope_digest': _digest(data['scope']),
               'observed_branch_digest': _digest(list(snapshot.order))}
    data['diagnostics']['cm_diagnostic_' + _digest(payload)] = payload


def capture_witnesses(store, snapshot=None):
    """Record only complete groups the host saw, before canonical replacement."""
    if not store.config.context_map_enabled and store._context_map is None:
        return
    data = _ensure(store)
    snap = snapshot or invariants.capture(store)
    validation = invariants.protocol(snap)
    invalid = {v.message_index for v in validation.violations
               if v.code != 'interrupted_tool_group'}
    witnessed = {tuple(w['source_uids']) for w in data['witnesses'].values()}
    branch = _digest(list(snap.order))
    for group in validation.groups:
        sources = list(group.source_uids)
        if (not group.complete or invalid.intersection(group.source_indices)
                or tuple(sources) in witnessed):
            continue
        key, record = _witness(store, snap, sources, 'canonical_group', branch)
        data['witnesses'][key] = record
        witnessed.add(tuple(sources))
    for uid, rec in snap.messages.items():
        if rec['role'] != 'tool' and not rec['envelope'].get('tool_calls') and (uid,) not in witnessed:
            try:
                invariants.source_groups(snap, [uid])
            except ValueError:
                continue
            key, record = _witness(store, snap, [uid], 'original', None)
            data['witnesses'][key] = record
            witnessed.add((uid,))
    # Accepted summary/body provenance is an independent retained derivation
    # witness, unlike guessing historical groups from matching call IDs.
    derivations = [s['derivation_source_uids'] for s in snap.summaries.values()]
    for rec in snap.messages.values():
        derivations.extend(p['derivation_source_uids'] for p in
                           [rec['body_compaction'], *rec['body_compaction_history']] if p is not None)
    for sources in derivations:
        for group in invariants.source_groups(snap, sources):
            group_uids = list(group.source_uids)
            if tuple(group_uids) not in witnessed:
                key, record = _witness(store, snap, group_uids, 'derivation_group', _digest(sources))
                data['witnesses'][key] = record
                witnessed.add(tuple(group_uids))
    _capture_diagnostic_observation(store, data, snap)


def _raw_ref(data, witness_id):
    witness = data['witnesses'][witness_id]
    existing = next((r for r in data['archive_refs'].values()
                     if r['source_uids'] == witness['source_uids']), None)
    if existing is not None:
        return existing['archive_ref']
    data['archive_seq'] += 1
    ref_id = f"cm_archive_{_digest(data['scope'])[:16]}_{data['archive_seq']:08d}"
    data['archive_refs'][ref_id] = {
        'archive_ref': ref_id, 'witness_id': witness_id,
        'source_uids': list(witness['source_uids']), 'scope_digest': _digest(data['scope']),
        'source_indices': list(witness['source_indices']),
        'envelope_digests': list(witness['envelope_digests']),
    }
    return ref_id


def _path_node(store, data, path, description='', keywords=()):
    if (not isinstance(path, (list, tuple)) or not path
            or len(path) > 5):
        raise ValueError('invalid_context_map_depth')
    for part in path:
        _text(part, 96, empty=False)
    _text(description, 256)
    words = _keywords(keywords)
    parent = data['nodes'][data['root_id']]
    for part in path:
        node = next((data['nodes'][c] for c in parent['child_ids']
                     if _norm(data['nodes'][c]['title']) == _norm(part)), None)
        if node is None:
            # Lowered creation limits do not invalidate saved ancestry.
            if len(path) > store.config.context_map_max_depth:
                raise ValueError('invalid_context_map_depth')
            node = _new_node(data, parent['node_id'], part)
        parent = node
    # Reuse without a metadata proposal does not erase earlier authored labels.
    if description:
        if description != parent['description']:
            _text(description, store.config.context_map_description_limit)
        parent['description'] = description
    if words:
        parent['keywords'] = words
    return parent


def _presentation_records(store):
    records = {}
    for summary in store.summaries.values():
        payload = summary.to_dict()
        key = 'cm_presentation_' + _digest(['summary', summary.uid, payload])
        records[key] = {'presentation_id': key, 'kind': 'summary', 'owner_uid': summary.uid,
                        'payload_digest': _digest(payload),
                        'source_uids': list(summary.source_message_ids),
                        'derivation_source_uids': list(summary.derivation_source_uids)}
    for uid, rec in store.messages.items():
        for payload in [rec.body_compaction, *rec.body_compaction_history]:
            if payload is not None:
                key = 'cm_presentation_' + _digest(['body', uid, payload])
                records[key] = {'presentation_id': key, 'kind': 'body', 'owner_uid': uid,
                                'payload_digest': _digest(payload),
                                'source_uids': list(payload['restore_group_uids']),
                                'derivation_source_uids': list(payload['derivation_source_uids'])}
    return records


def _refresh_derived(store, data, before):
    def visit(node_id):
        node = data['nodes'][node_id]
        sources = {u for ref in node['archive_refs'] for u in data['archive_refs'][ref]['source_uids']}
        sources.update(u for ref in node['diagnostic_refs']
                       for u in data['diagnostics'][ref]['uncertified_source_uids'])
        for child in node['child_ids']:
            sources.update(visit(child))
        node['provenance_ids'] = sorted(sources, key=lambda u: (store.messages[u].index, u))
        previous = before['nodes'].get(node_id)
        # Retained envelopes have validated immutable identities. Calibration
        # alone must not rewrite their accepted metadata or an issued cursor.
        # Reuse the host's preimage, never an imported estimate proposal.
        if previous is not None and previous['provenance_ids'] == node['provenance_ids']:
            node['token_estimate'] = previous['token_estimate']
        else:
            node['token_estimate'] = sum(store.counter.count_message(store.messages[u].envelope)
                                         for u in node['provenance_ids'])
        return sources
    visit(data['root_id'])


def _touch_changed_nodes(data, before):
    # Modification time records authored/topology/provenance changes, never a
    # no-op read or hidden witness capture. Access has separate host metadata.
    ignored = {'updated_at', 'last_accessed_at', 'access_count'}
    for node_id, node in data['nodes'].items():
        previous = before['nodes'].get(node_id)
        if previous is not None and any(node[k] != previous[k] for k in NODE_FIELDS - ignored):
            node['updated_at'] = utc_now_iso()


def sync_map(store, *, diagnostic=False):
    if not store.config.context_map_enabled and store._context_map is None:
        return
    data = _ensure(store)
    before = deepcopy(data)
    capture_witnesses(store)
    archived = {u for u, rec in store.messages.items() if rec.archived}
    presentations = _presentation_records(store)
    derived = {u for p in presentations.values() for u in p['derivation_source_uids']}
    wanted = archived | derived
    covered = {u for w in data['witnesses'].values() for u in w['source_uids']}
    audited = {u for d in data['diagnostics'].values() for u in d['uncertified_source_uids']}
    if not derived.issubset(covered) or not archived.issubset(covered | audited):
        if diagnostic:
            data['recovery_required'] = 'context_map_provenance_unverifiable'
            return
        raise RecoveryRequired('context_map_provenance_unverifiable')
    data['recovery_required'] = None
    for key, witness in sorted(data['witnesses'].items(),
                               key=lambda item: (item[1]['source_indices'], item[1]['source_uids'])):
        if not wanted.intersection(witness['source_uids']):
            continue
        ref = _raw_ref(data, key)
        if not any(ref in n['archive_refs'] for n in data['nodes'].values()):
            indices = witness['source_indices']
            historic = not set(witness['source_uids']).issubset(store.order)
            branch = 'History' if historic else 'Archives'
            title = f"Source {indices[0]}–{indices[-1]}"
            path = [f'{branch}: {title}'] if store.config.context_map_max_depth == 1 else [branch, title]
            node = _path_node(store, data, path)
            node['archive_refs'].append(ref)
    for diagnostic_id in data['diagnostics']:
        if not any(diagnostic_id in n['diagnostic_refs'] for n in data['nodes'].values()):
            title = 'Observation ' + diagnostic_id.removeprefix('cm_diagnostic_')
            path = [f'Diagnostics: {title}'] if store.config.context_map_max_depth == 1 else ['Diagnostics', title]
            node = _path_node(store, data, path)
            node['diagnostic_refs'].append(diagnostic_id)
    data['presentations'] = presentations
    for node in data['nodes'].values():
        node['presentation_refs'] = [r for r in node['presentation_refs'] if r in presentations]
    for presentation_id, presentation in presentations.items():
        sources = set(presentation['derivation_source_uids'])
        refs = [r for r, rec in data['archive_refs'].items() if set(rec['source_uids']).issubset(sources)]
        node = next((n for n in data['nodes'].values()
                     if n['presentation_refs'] and presentation_id in n['presentation_refs']), None)
        if node is None:
            node = next(n for n in data['nodes'].values() if any(r in n['archive_refs'] for r in refs))
        for ref in refs:
            if ref not in node['archive_refs']:
                node['archive_refs'].append(ref)
        if presentation_id not in node['presentation_refs']:
            node['presentation_refs'].append(presentation_id)
    for node in data['nodes'].values():
        # A presentation can belong to several source-complete nodes. Preserve
        # each association and its existing summary order during no-op syncs.
        summaries = [presentations[r]['owner_uid'] for r in node['presentation_refs']
                     if presentations[r]['kind'] == 'summary']
        node['summary_refs'] = [uid for uid in node['summary_refs'] if uid in summaries]
        for uid in summaries:
            if uid not in node['summary_refs']:
                node['summary_refs'].append(uid)
    _refresh_derived(store, data, before)
    _touch_changed_nodes(data, before)
    # Hidden witness capture and access do not churn the visible topology revision.
    keys = ('nodes', 'archive_refs', 'presentations', 'diagnostics', 'recovery_required')
    if any(data[k] != before[k] for k in keys):
        data['revision'] += 1
        store._dirty = True


def validate_map(store, data):
    """Pure full validation, called by the store without reentering facade guards."""
    validate_json(data, '/context_map')
    if type(data) is not dict or set(data) != MAP_FIELDS or type(data['version']) is not int or data['version'] != 1:
        raise ValueError('invalid_context_map_schema')
    if data['scope'] != _scope(store):
        raise ValueError('foreign_context_map_scope')
    if data['recovery_required'] is not None:
        if data['recovery_required'] != 'context_map_provenance_unverifiable':
            raise ValueError('invalid_context_map_recovery')
        raise RecoveryRequired(data['recovery_required'])
    for name in ('revision', 'node_seq', 'archive_seq'):
        _integer(data[name])
    for name in ('nodes', 'archive_refs', 'witnesses', 'presentations', 'diagnostics'):
        if type(data[name]) is not dict:
            raise ValueError('invalid_context_map_registry')
    prefix = _digest(data['scope'])[:16]
    snap = invariants.capture(store)
    for diagnostic_id, record in data['diagnostics'].items():
        if type(record) is not dict or set(record) != {
                'observed_source_uids', 'envelope_digests', 'uncertified_source_uids',
                'scope_digest', 'observed_branch_digest'}:
            raise ValueError('invalid_context_map_diagnostic')
        _strings(record['observed_source_uids'], nonempty=True)
        _strings(record['uncertified_source_uids'], nonempty=True)
        branch = record['observed_source_uids']
        if (any(u not in snap.messages for u in branch)
                or [snap.messages[u]['index'] for u in branch] != list(range(len(branch)))
                or record['scope_digest'] != _digest(data['scope'])
                or record['observed_branch_digest'] != _digest(branch)
                or record['envelope_digests'] != [envelope_digest(snap.messages[u]['envelope']) for u in branch]
                or diagnostic_id != 'cm_diagnostic_' + _digest(record)):
            raise ValueError('invalid_context_map_diagnostic')
        observed = replace(snap, order=tuple(branch), by_index=dict(enumerate(branch)))
        if record['uncertified_source_uids'] != _uncertified_sources(observed):
            raise ValueError('invalid_context_map_diagnostic_sources')
    witnessed_groups = set()
    for key, witness in data['witnesses'].items():
        if type(witness) is not dict or set(witness) != {
                'source_uids', 'envelope_digests', 'source_indices', 'scope_digest',
                'authority', 'observed_branch_digest'} or key != _digest(witness):
            raise ValueError('invalid_context_map_witness')
        if witness['scope_digest'] != _digest(data['scope']):
            raise ValueError('foreign_context_map_witness')
        if witness['authority'] not in {'canonical_group', 'derivation_group', 'original'}:
            raise ValueError('invalid_context_map_witness_authority')
        branch = witness['observed_branch_digest']
        if ((witness['authority'] == 'original' and branch is not None)
                or (witness['authority'] != 'original' and
                    (not isinstance(branch, str) or len(branch) != 64
                     or any(c not in '0123456789abcdef' for c in branch)))):
            raise ValueError('invalid_context_map_branch_witness')
        _strings(witness['source_uids'], nonempty=True)
        _indices(witness['source_indices'])
        groups = invariants.source_groups(snap, witness['source_uids'])
        if groups:
            if len(groups) != 1 or list(groups[0].source_uids) != witness['source_uids'] or witness['authority'] == 'original':
                raise ValueError('invalid_context_map_group_witness')
        elif len(witness['source_uids']) != 1 or witness['authority'] != 'original':
            raise ValueError('invalid_context_map_original_witness')
        if (witness['envelope_digests'] != [envelope_digest(store.messages[u].envelope) for u in witness['source_uids']]
                or witness['source_indices'] != [store.messages[u].index for u in witness['source_uids']]):
            raise ValueError('changed_context_map_source')
        membership = tuple(witness['source_uids'])
        if membership in witnessed_groups:
            raise ValueError('duplicate_context_map_witness')
        witnessed_groups.add(membership)
    raw_membership = set()
    for ref_id, ref in data['archive_refs'].items():
        if type(ref) is not dict or set(ref) != {'archive_ref', 'witness_id', 'source_uids',
                'scope_digest', 'source_indices', 'envelope_digests'}:
            raise ValueError('invalid_archive_ref_schema')
        number = ref_id.removeprefix(f'cm_archive_{prefix}_')
        if (not number.isascii() or not number.isdecimal() or int(number) < 1
                or int(number) > data['archive_seq'] or ref_id != f'cm_archive_{prefix}_{int(number):08d}'
                or ref['archive_ref'] != ref_id):
            raise ValueError('invalid_archive_ref_allocation')
        witness = data['witnesses'].get(ref['witness_id'])
        _indices(ref['source_indices'])
        if witness is None or ref['scope_digest'] != _digest(data['scope']):
            raise ValueError('invalid_archive_ref_witness')
        for field in ('source_uids', 'source_indices', 'envelope_digests'):
            if ref[field] != witness[field]:
                raise ValueError('corrupt_archive_ref')
        group = tuple(ref['source_uids'])
        if group in raw_membership:
            raise ValueError('duplicate_archive_ref')
        raw_membership.add(group)
    if data['presentations'] != _presentation_records(store):
        raise ValueError('invalid_context_map_presentation_provenance')
    root_id = data['root_id']
    if root_id not in data['nodes']:
        raise ValueError('missing_context_map_root')
    seen, reached_refs, reached_presentations, reached_diagnostics = set(), set(), set(), set()
    def visit(node_id, parent_id, depth):
        if node_id in seen or depth > 5:
            raise ValueError('context_map_cycle_or_depth')
        seen.add(node_id)
        node = data['nodes'].get(node_id)
        if type(node) is not dict or set(node) != NODE_FIELDS:
            raise ValueError('invalid_context_map_node')
        number = node_id.removeprefix(f'cm_node_{prefix}_')
        if (not number.isascii() or not number.isdecimal() or int(number) < 1
                or int(number) > data['node_seq'] or node_id != f'cm_node_{prefix}_{int(number):08d}'
                or node['node_id'] != node_id or node['parent_id'] != parent_id):
            raise ValueError('invalid_context_map_node_allocation_or_parent')
        _text(node['title'], 96, empty=False)
        _text(node['description'], 256)
        _keywords(node['keywords'])
        for field in ('child_ids', 'archive_refs', 'summary_refs', 'presentation_refs', 'diagnostic_refs', 'provenance_ids'):
            _strings(node[field])
        for field in ('token_estimate', 'access_count'):
            _integer(node[field])
        _timestamp(node['created_at']); _timestamp(node['updated_at'])
        _timestamp(node['last_accessed_at'], nullable=True)
        if (node['access_count'] == 0) != (node['last_accessed_at'] is None):
            raise ValueError('invalid_context_map_access')
        labels = [_norm(data['nodes'][c]['title']) for c in node['child_ids'] if c in data['nodes']]
        if len(labels) != len(set(labels)):
            raise ValueError('duplicate_context_map_path')
        if type(node['links']) is not list or len(node['links']) > 16:
            raise ValueError('invalid_context_map_links')
        for link in node['links']:
            if (type(link) is not dict or set(link) != {'relation', 'node_id'}
                    or link['relation'] not in {'correction', 'supersedes', 'unresolved'}
                    or link['node_id'] not in data['nodes']):
                raise ValueError('invalid_context_map_link')
        sources = set()
        for ref in node['archive_refs']:
            if ref not in data['archive_refs']:
                raise ValueError('dangling_archive_ref')
            reached_refs.add(ref)
            sources.update(data['archive_refs'][ref]['source_uids'])
        expected_summaries = set()
        for presentation_id in node['presentation_refs']:
            presentation = data['presentations'].get(presentation_id)
            if presentation is None:
                raise ValueError('dangling_context_map_presentation')
            reached_presentations.add(presentation_id)
            if not set(presentation['derivation_source_uids']).issubset(sources):
                raise ValueError('incomplete_context_map_presentation_sources')
            if presentation['kind'] == 'summary':
                expected_summaries.add(presentation['owner_uid'])
        if set(node['summary_refs']) != expected_summaries:
            raise ValueError('invalid_context_map_summary_refs')
        for diagnostic_id in node['diagnostic_refs']:
            if diagnostic_id not in data['diagnostics']:
                raise ValueError('dangling_context_map_diagnostic')
            reached_diagnostics.add(diagnostic_id)
            sources.update(data['diagnostics'][diagnostic_id]['uncertified_source_uids'])
        if node_id != root_id and not node['child_ids'] and not sources:
            raise ValueError('orphan_context_map_leaf')
        for child in node['child_ids']:
            sources.update(visit(child, node_id, depth + 1))
        expected_sources = sorted(sources, key=lambda u: (store.messages[u].index, u))
        if node['provenance_ids'] != expected_sources:
            raise ValueError('invalid_context_map_provenance')
        return sources
    covered = visit(root_id, None, 0)
    if (seen != set(data['nodes']) or reached_refs != set(data['archive_refs'])
            or reached_presentations != set(data['presentations'])
            or reached_diagnostics != set(data['diagnostics'])
            or not {u for u, r in store.messages.items() if r.archived}.issubset(covered)):
        raise ValueError('incomplete_context_map_coverage')


@_accepted_mutation
def _map_mutation(store, operation, *args, **kwargs):
    data = _ensure(store)
    before = deepcopy(data)
    result = operation(store, data, *args, **kwargs)
    _refresh_derived(store, data, before)
    _touch_changed_nodes(data, before)
    if data != before:
        data['revision'] += 1
        store._dirty = True
    return result


class ContextMapIndex:
    def __init__(self, store):
        self.store = store

    def _read(self, *, inspection=False):
        self.store.config.validate_context_map()
        self.store.validate_accepted(inspection=inspection)
        if self.store._context_map is None:
            raise ValueError('context_map_unavailable')
        return self.store._context_map

    def create_node(self, path, source_uids, *, description='', keywords=()):
        self._read()
        def create(store, data):
            _strings(source_uids, nonempty=True)
            invariants.source_groups(invariants.capture(store), source_uids)
            available = [key for key, witness in data['witnesses'].items()
                         if set(witness['source_uids']).issubset(source_uids)]
            covered = {u for key in available for u in data['witnesses'][key]['source_uids']}
            if covered != set(source_uids):
                raise ValueError('context_map_sources_unproved')
            node = _path_node(store, data, path, description, keywords)
            for key in sorted(available, key=lambda k: data['witnesses'][k]['source_indices']):
                ref = _raw_ref(data, key)
                if ref not in node['archive_refs']:
                    node['archive_refs'].append(ref)
            return node['node_id']
        node_id = _map_mutation(self.store, create)
        return self.get_node(node_id)

    def attach_archive_ref(self, node_id, archive_ref):
        self._read()
        def attach(store, data):
            if node_id not in data['nodes'] or archive_ref not in data['archive_refs']:
                raise ValueError('unknown_context_node_or_archive_ref')
            if archive_ref not in data['nodes'][node_id]['archive_refs']:
                data['nodes'][node_id]['archive_refs'].append(archive_ref)
        _map_mutation(self.store, attach)
        return self.get_node(node_id)

    def update_node(self, node_id, *, title=None, description=None, keywords=None, links=None):
        self._read()
        def update(store, data):
            if node_id not in data['nodes']:
                raise ValueError('unknown_context_node')
            node = data['nodes'][node_id]
            if title is not None:
                node['title'] = _text(title, 96, empty=False)
            if description is not None:
                _text(description, 256)
                if description != node['description']:
                    _text(description, store.config.context_map_description_limit)
                node['description'] = description
            if keywords is not None:
                node['keywords'] = _keywords(keywords)
            if links is not None:
                node['links'] = deepcopy(links)
        _map_mutation(self.store, update)
        return self.get_node(node_id)

    def get_node(self, node_id):
        data = self._read()
        if not isinstance(node_id, str) or node_id not in data['nodes']:
            raise ValueError('unknown_context_node')
        return ContextMapNode._from_record(data['nodes'][node_id])

    def search_nodes(self, query, max_results=_OMITTED_LIMIT):
        data = self._read()
        from .lookup import rank_nodes, result_limit
        limit = result_limit(self.store.config, max_results)
        return [ContextMapNode._from_record(r['node']) for r in rank_nodes(data, query)[:limit]]

    def _record_lookup_access(self, node_ids):
        self._read()
        def access(store, data):
            _strings(node_ids)
            if any(key not in data['nodes'] or key == data['root_id'] for key in node_ids):
                raise ValueError('unknown_context_node')
            stamp = utc_now_iso()
            for key in node_ids:
                data['nodes'][key]['access_count'] += 1
                data['nodes'][key]['last_accessed_at'] = stamp
        if node_ids:
            _map_mutation(self.store, access)

    def render_snapshot(self):
        # Structural overview only; provider-framed progressive disclosure and
        # exact component admission are separate from this off-prompt API.
        data = self._read()
        root = data['nodes'][data['root_id']]
        shown = root['child_ids'][:self.store.config.context_map_max_results]
        return deepcopy({'revision': data['revision'], 'root_id': data['root_id'],
                         'nodes': [data['nodes'][n] for n in [data['root_id'], *shown]],
                         'omitted_children': len(root['child_ids']) - len(shown)})

    def validate_refs(self, archive_refs=None, *, require_current=False):
        data = self._read()
        if type(require_current) is not bool:
            raise ValueError('invalid_context_map_current_requirement')
        ids = list(data['archive_refs']) if archive_refs is None else archive_refs
        _strings(ids)
        from ..policies import PolicyViolation
        result = {}
        current_groups = {tuple(g.source_uids) for g in self.store.groups() if g.complete}
        for ref_id in ids:
            ref = data['archive_refs'].get(ref_id)
            if ref is None:
                raise PolicyViolation('unknown_archive_ref', 'Unknown archive reference')
            sources = ref['source_uids']
            eligible = set(sources).issubset(self.store.order)
            if any(self.store.messages[u].role == 'tool' or self.store.messages[u].envelope.get('tool_calls') for u in sources):
                eligible = eligible and tuple(sources) in current_groups
            if require_current and not eligible:
                raise PolicyViolation('historic_branch_conflict', 'Reference belongs to preserved historic context', sources)
            result[ref_id] = deepcopy(dict(ref, current_branch=eligible))
        return {'ok': True, 'refs': result}

    def to_dict(self):
        return deepcopy(self._read(inspection=True))

    @classmethod
    def from_dict(cls, data, *, store):
        index = cls(store)
        existing = index._read()
        candidate = deepcopy(data)
        validate_map(store, candidate)
        # Import cannot mint a witness or operational identity. Full checkpoint
        # decoding has its own temporary-store adoption path.
        for key in ('scope', 'revision', 'node_seq', 'archive_seq', 'root_id', 'witnesses', 'archive_refs', 'presentations', 'diagnostics'):
            if candidate[key] != existing[key]:
                raise ValueError('context_map_import_identity_changed')
        if set(candidate['nodes']) != set(existing['nodes']):
            raise ValueError('context_map_import_identity_changed')
        def ancestry(nodes, node_id):
            # Both complete maps have already passed bounded acyclic topology
            # validation. A changed ancestor affects every descendant's path.
            path = []
            while node_id is not None:
                path.append(node_id)
                node_id = nodes[node_id]['parent_id']
            return tuple(path)
        for node_id, node in candidate['nodes'].items():
            previous = existing['nodes'][node_id]
            if node['description'] != previous['description']:
                _text(node['description'], store.config.context_map_description_limit)
            proposed_path = ancestry(candidate['nodes'], node_id)
            if (proposed_path != ancestry(existing['nodes'], node_id)
                    and len(proposed_path) - 1 > store.config.context_map_max_depth):
                raise ValueError('invalid_context_map_depth')
            for field in ('created_at', 'updated_at', 'last_accessed_at', 'access_count'):
                if node[field] != existing['nodes'][node_id][field]:
                    raise ValueError('context_map_import_host_metadata_changed')
        def adopt(receiver, current):
            current.clear()
            current.update(candidate)
        _map_mutation(store, adopt)
        return cls(store)
