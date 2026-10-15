"""Bounded metadata lookup. Caller arguments cannot choose attribution."""
from ..retrieval.context_map import _OMITTED_LIMIT
from ..policies import PolicyViolation


def tool_lookup(engine, args):
    if type(args) is not dict or set(args) - {'query', 'max_results', 'cursor'}:
        return {'ok': False, 'error': 'invalid_lookup_arguments'}
    try:
        return engine.apply_lookup(args.get('query'), max_results=args.get('max_results', _OMITTED_LIMIT),
                                   cursor=args.get('cursor'))
    except ValueError as error:
        code = str(error)
        if code not in ('context_map_disabled', 'invalid_lookup_query', 'invalid_context_map_result_limit',
                        'invalid_lookup_cursor', 'stale_lookup_cursor'):
            code = 'context_map_invalid'
        raise PolicyViolation(code, str(error)) from None
