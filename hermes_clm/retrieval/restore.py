"""Pure current-reference resolution for the existing restore transaction."""
from ..policies import PolicyViolation

OMITTED = object()


def plan_reference_restore(engine, archive_refs, reason=OMITTED):
    if not engine.config.context_map_enabled:
        raise PolicyViolation('context_map_disabled', 'Archive-reference restoration is disabled')
    if (type(archive_refs) is not list or not 1 <= len(archive_refs) <= 5
            or any(type(ref) is not str or not 1 <= len(ref) <= 96 for ref in archive_refs)):
        raise PolicyViolation('invalid_archive_refs', 'Supply one through five exact archive references')
    try:
        for ref in archive_refs:
            ref.encode('utf-8')
    except UnicodeError:
        raise PolicyViolation('invalid_archive_refs', 'Archive references must be valid UTF-8') from None
    if reason is not OMITTED:
        try:
            if type(reason) is not str or not reason.strip() or len(reason) > 2000:
                raise ValueError()
            reason.encode('utf-8')
        except (ValueError, UnicodeError):
            raise PolicyViolation('invalid_restore_reason', 'Reason must be nonempty UTF-8 text of at most 2000 characters') from None
    refs = list(dict.fromkeys(archive_refs))
    index = engine.store.context_map_index
    validated = index.validate_refs(refs, require_current=True)['refs']
    sources = sorted({uid for ref in validated.values() for uid in ref['source_uids']},
                     key=lambda uid: engine.store.get(uid).index)
    selected = set(refs)
    data = index.to_dict()
    nodes = sorted(key for key, node in data['nodes'].items()
                   if selected.intersection(node['archive_refs']))
    provenance = []
    for ref in refs:
        for receipt in reversed(engine._lookup_receipts):
            if receipt['scope'] != data['scope']:
                continue
            candidate = next((row for row in receipt['candidates']
                if row['archive_ref'] == ref and row['current_branch']), None)
            if candidate is not None:
                provenance.append(dict(archive_ref=ref, lookup_id=receipt['lookup_id'],
                    query=receipt['query'], normalized_query=receipt['normalized_query'],
                    node_id=candidate['node_id'], rank=candidate['rank']))
                break
    attributed = {row['archive_ref'] for row in provenance}
    return sources, dict(archive_refs=refs, source_node_ids=nodes,
                         reason=None if reason is OMITTED else reason,
                         lookup_provenance=provenance,
                         lookup_ids=list(dict.fromkeys(row['lookup_id'] for row in provenance)),
                         unattributed_archive_refs=[ref for ref in refs if ref not in attributed],
                         lookup_provenance_scope='admitted_recent_session_lookup')
