"""Parity evidence and a provider-bound final-payload counter.

Evidence validation is not evidence collection. Offline fixture records remain
explicitly unqualified; only independently collected live-provider records may
make a live counter exact. A backend must attest current identity and endpoint
availability even on a cache hit.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from .provider_budget import (ProviderIdentity, PromptCount, effective_request_body,
                              prompt_fingerprint, _json_value)


class CounterError(RuntimeError):
    def __init__(self, code, message):
        self.code=code
        super().__init__(f'{code}: {message}')


def _sha256(value):
    return type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def _coverage(body):
    features=set();messages=body.get('messages',[])
    if type(messages) is not list:
        raise ValueError('parity payload must contain a messages list')
    if body.get('tools'):features.add('tools')
    if body.get('chat_template_kwargs'):features.add('template_options')
    if len(messages)>=6:features.add('resumed')
    for message in messages:
        if type(message) is not dict:raise ValueError('invalid parity message')
        content=message.get('content')
        if content is None:features.add('null_content')
        if type(content) is list:features.add('text_blocks')
        if type(content) is str:
            if any(ord(char)>127 for char in content):features.add('unicode')
            if '```' in content:features.add('code')
        if message.get('reasoning_content') or message.get('reasoning_details'):features.add('reasoning')
        if len(message.get('tool_calls') or [])>1:features.add('parallel_tools')
        if message.get('role')=='tool':features.add('tool_results')
    return features


_REQUIRED_COVERAGE=frozenset({'tools','template_options','resumed','null_content',
    'text_blocks','unicode','code','reasoning','parallel_tools','tool_results'})


@dataclass(frozen=True)
class ParityCertificate:
    identity: ProviderIdentity
    prompt_fingerprints: tuple[str,...]
    bos_delta: int
    evidence_sha256: str
    evidence_origin: str
    coverage: frozenset[str]

    def __post_init__(self):
        if not isinstance(self.identity,ProviderIdentity):raise ValueError('missing parity identity')
        if (type(self.prompt_fingerprints) is not tuple or len(set(self.prompt_fingerprints))<30
                or any(not _sha256(value) for value in self.prompt_fingerprints)):
            raise ValueError('at least30 distinct complete parity payloads are required')
        if type(self.bos_delta) is not int or self.bos_delta not in (-1,0,1):
            raise ValueError('BOS convention must be a fixed difference of at most one token')
        if not _sha256(self.evidence_sha256):raise ValueError('missing parity evidence hash')
        if self.evidence_origin not in ('offline_fixture','live_provider'):
            raise ValueError('parity evidence origin must be explicit')
        if type(self.coverage) is not frozenset or not _REQUIRED_COVERAGE.issubset(self.coverage):
            raise ValueError('parity payloads do not cover the required representative shapes')

    @property
    def live_qualified(self):return self.evidence_origin=='live_provider'
    @property
    def payload_count(self):return len(self.prompt_fingerprints)
    @property
    def bos_adjustment(self):return max(0,self.bos_delta)

    @classmethod
    def from_observations(cls,identity,observations,*,bos_delta,evidence_sha256,evidence_origin):
        fingerprints=[];coverage=set()
        for row in observations:
            if row.get('identity')!=identity:raise ValueError('parity identity changed')
            ids=row.get('token_ids');usage=row.get('provider_prompt_tokens')
            if type(ids) is not tuple or not ids or any(type(token) is not int or token<0 for token in ids):
                raise ValueError('parity evidence must retain exact token IDs')
            if type(usage) is not int or usage<0 or usage-len(ids)!=bos_delta:
                raise ValueError('prompt usage does not match the fixed BOS convention')
            body=effective_request_body(row['request'])
            if body.get('model')!=identity.model:raise ValueError('parity model changed')
            fingerprints.append(prompt_fingerprint(row['request'],identity));coverage.update(_coverage(body))
        return cls(identity,tuple(fingerprints),bos_delta,evidence_sha256,evidence_origin,frozenset(coverage))


class ExactRequestCounter:
    """Service object; do not deep-copy or persist with mutable conversation state.

    The backend supplies check_identity_and_availability() and
    tokenize_request(complete_body). These are tokenization operations only,
    never inference requests. Qualification records and current runtime identity
    are mandatory; response usage is never used to fit a request's count.
    """
    def __init__(self,backend,certificate: ParityCertificate,*,cache_entries=16):
        if not isinstance(certificate,ParityCertificate):raise ValueError('parity certificate required')
        if type(cache_entries) is not int or not 1<=cache_entries<=128:raise ValueError('cache_entries must be1..128')
        self.backend=backend;self.certificate=certificate;self.cache_entries=cache_entries
        self._cache: OrderedDict[str,PromptCount]=OrderedDict()

    @property
    def cache_size(self):return len(self._cache)

    def _check(self,expected):
        try:current=self.backend.check_identity_and_availability()
        except CounterError:
            self._cache.clear()
            raise
        except Exception as exc:
            self._cache.clear()
            raise CounterError('tokenizer_unavailable',str(exc)) from exc
        if current!=expected or current!=self.certificate.identity:
            self._cache.clear()
            raise CounterError('provider_identity_changed','Current provider does not match qualified model/tokenizer/template/build/ceiling')

    def count_final_request(self,request: dict,identity: ProviderIdentity) -> PromptCount:
        if request.get('extra_headers'):
            raise CounterError('unsupported_request_headers',
                'Per-request headers are not propagated to the certified count routes')
        body=effective_request_body(request)
        if body.get('model')!=identity.model:
            raise CounterError('provider_identity_changed','Requested model differs from qualified identity')
        self._check(identity)
        fingerprint=prompt_fingerprint(request,identity)
        if fingerprint in self._cache:
            self._cache.move_to_end(fingerprint)
            return self._cache[fingerprint]
        try:
            ids=self.backend.tokenize_request(body)
            if type(ids) is list:ids=tuple(ids)
            count=PromptCount(ids,self.certificate.live_qualified,fingerprint,identity,self.certificate.bos_adjustment)
        except CounterError:raise
        except Exception as exc:raise CounterError('tokenizer_unavailable',str(exc)) from exc
        self._check(identity)
        self._cache[fingerprint]=count
        while len(self._cache)>self.cache_entries:self._cache.popitem(last=False)
        return count


class HttpCounterTransport:
    """Small HTTP client restricted to the four non-inference llama routes.

    Credentials are supplied by the owning profile. Redirects are forbidden so
    neither credentials nor a qualified request can silently change providers.
    No environment proxy or environment credential discovery is performed.
    """
    _routes = frozenset({('GET', '/props'), ('POST', '/apply-template'),
                        ('POST', '/tokenize'),
                        ('POST', '/v1/chat/completions/input_tokens')})

    def __init__(self, base_url, *, api_key=None, timeout=10.0):
        from urllib.parse import urlsplit, urlunsplit
        parsed = urlsplit(base_url)
        if (parsed.scheme not in ('http', 'https') or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError('counter base_url must be an HTTP(S) origin and optional path')
        path = parsed.path.rstrip('/')
        if path.endswith('/v1'):
            path = path[:-3]
        self.base_url = urlunsplit((parsed.scheme, parsed.netloc, path, '', ''))
        if type(timeout) not in (int, float) or not 0 < timeout <= 60:
            raise ValueError('counter timeout must be positive and at most 60 seconds')
        if api_key is not None and (type(api_key) is not str or '\n' in api_key or '\r' in api_key):
            raise ValueError('invalid counter credential')
        self._api_key = api_key
        self.timeout = timeout

    def request_json(self, method, path, body=None):
        import json
        from urllib.request import (Request, build_opener, ProxyHandler,
                                    HTTPRedirectHandler)
        if (method, path) not in self._routes:
            raise ValueError('only a supported count endpoint may be called')
        if method == 'GET' and body is not None:
            raise ValueError('GET count endpoint does not accept a body')

        class NoRedirect(HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None

        headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}
        if self._api_key:
            headers['Authorization'] = 'Bearer ' + self._api_key
        data = None if body is None else json.dumps(
            _json_value(body), ensure_ascii=False, separators=(',', ':'),
            allow_nan=False).encode('utf-8')
        request = Request(self.base_url + path, data=data, headers=headers, method=method)
        with build_opener(ProxyHandler({}), NoRedirect()).open(request, timeout=self.timeout) as response:
            payload = response.read(64 * 1024 * 1024 + 1)
        if len(payload) > 64 * 1024 * 1024:
            raise CounterError('tokenizer_unavailable', 'Counter response exceeds 64 MiB')
        result = json.loads(payload)
        if type(result) is not dict:
            raise CounterError('tokenizer_unavailable', 'Counter response must be a JSON object')
        return result


class LlamaServerTokenizer:
    """Same-parser rendering and token IDs, checked against llama's count route.

    identity_reader is a trusted host attestation service, not a value inferred
    from /props. It must identify the running process and model/tokenizer/server
    files. Props then independently checks the effective runtime configuration.
    This adapter supports text requests only; parity cannot certify media tokens.
    """
    def __init__(self, identity, identity_reader, transport):
        if not isinstance(identity, ProviderIdentity) or not callable(identity_reader):
            raise ValueError('provider identity and host attestation reader are required')
        self.identity = identity
        self.identity_reader = identity_reader
        self.transport = transport

    def _request(self, method, path, body=None):
        try:
            result = self.transport.request_json(method, path, body)
            if type(result) is not dict:
                raise ValueError('endpoint response must be an object')
            return result
        except CounterError:
            raise
        except Exception as exc:
            raise CounterError('tokenizer_unavailable', f'{path}: {exc}') from exc

    def check_identity_and_availability(self):
        import hashlib
        try:
            current = self.identity_reader()
        except Exception as exc:
            raise CounterError('tokenizer_unavailable', 'Host attestation unavailable') from exc
        if current != self.identity:
            raise CounterError('provider_identity_changed', 'Host attestation changed')
        props = self._request('GET', '/props')
        template = props.get('chat_template')
        settings = props.get('default_generation_settings')
        if (props.get('model_alias') != current.model
                or type(template) is not str
                or hashlib.sha256(template.encode('utf-8')).hexdigest() != current.template_sha256
                or props.get('build_info') != current.build_id
                or type(settings) is not dict
                or type(settings.get('n_ctx')) is not int
                or settings['n_ctx'] != current.context_tokens):
            raise CounterError('provider_identity_changed', 'Runtime props changed')
        # A props response alone cannot prove that every required route is up.
        self.tokenize_request({'model': current.model,
                               'messages': [{'role': 'user', 'content': 'count health'}]})
        return current

    def tokenize_request(self, body):
        body = _json_value(body)
        if type(body) is not dict or body.get('model') != self.identity.model:
            raise CounterError('provider_identity_changed', 'Requested model changed')
        messages = body.get('messages')
        if type(messages) is not list or not messages:
            raise CounterError('unsupported_content', 'A nonempty text message list is required')
        if any(key in body for key in ('audio', 'images', 'videos', 'input_audio')):
            raise CounterError('unsupported_content', 'Media input is not certified')
        for message in messages:
            if type(message) is not dict:
                raise CounterError('unsupported_content', 'Invalid text message')
            content = message.get('content')
            if content is not None and type(content) is not str:
                if type(content) is not list or any(
                    type(block) is not dict or block.get('type') != 'text'
                    or type(block.get('text')) is not str for block in content
                ):
                    raise CounterError('unsupported_content', 'Only text content blocks are certified')
            if any(key in message for key in ('audio', 'images', 'videos', 'input_audio')):
                raise CounterError('unsupported_content', 'Media message is not certified')
        rendered = self._request('POST', '/apply-template', body).get('prompt')
        if type(rendered) is not str:
            raise CounterError('tokenizer_unavailable', 'Template endpoint omitted rendered prompt')
        ids = self._request('POST', '/tokenize', {
            'content': rendered, 'add_special': True, 'parse_special': True}).get('tokens')
        if type(ids) is not list or not ids or any(type(token) is not int or token < 0 for token in ids):
            raise CounterError('tokenizer_unavailable', 'Tokenizer did not return valid token IDs')
        total = self._request('POST', '/v1/chat/completions/input_tokens', body).get('input_tokens')
        if type(total) is not int or total != len(ids):
            raise CounterError('tokenizer_parity_failed', 'Same-parser count disagrees with token IDs')
        return tuple(ids)
