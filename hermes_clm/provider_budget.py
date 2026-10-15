"""Final provider-payload identity and exact-budget policy.

No text heuristic in this module can produce an exact PromptCount. A served
renderer/tokenizer and its parity evidence must supply the token sequence.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any


def _integer(value, name, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def _json_value(value, depth=0):
    """Copy only JSON primitives; never execute arbitrary copy/encoder hooks."""
    if depth > 128:
        raise ValueError('request nesting exceeds 128 levels')
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not (-float('inf') < value < float('inf')):
            raise ValueError('nonfinite request number')
        return value
    if type(value) is list:
        return [_json_value(item, depth+1) for item in value]
    if type(value) is dict and all(type(key) is str for key in value):
        return {key: _json_value(item, depth+1) for key,item in value.items()}
    raise ValueError('final request must contain only JSON values and string keys')


def _digest(value):
    # Preserve field order: a template may iterate mappings in insertion order.
    encoded=json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ProviderIdentity:
    provider: str
    base_url: str
    model: str
    model_sha256: str
    tokenizer_sha256: str
    template_sha256: str
    server_sha256: str
    build_id: str
    instance_id: str
    preprocessing_sha256: str
    context_tokens: int

    def __post_init__(self):
        _integer(self.context_tokens, 'context_tokens', minimum=1)
        for name,value in vars(self).items():
            if name=='context_tokens':
                continue
            if type(value) is not str or not value.strip():
                raise ValueError(f'{name} must be a nonempty string')
            if name.endswith('_sha256') and (len(value)!=64 or any(c not in '0123456789abcdef' for c in value)):
                raise ValueError(f'{name} must be a lowercase SHA256 digest')

    @property
    def fingerprint(self):
        return _digest(asdict(self))


@dataclass(frozen=True)
class BudgetLimits:
    provider_ceiling: int = 131072
    active_target: int = 32768
    reserve: int = 4096
    output_tokens: int = 4096
    safety_margin: int = 1024

    def __post_init__(self):
        for name in ('provider_ceiling','active_target','output_tokens'):
            _integer(getattr(self,name),name,minimum=1)
        for name in ('reserve','safety_margin'):
            _integer(getattr(self,name),name)
        if self.reserve >= self.active_target or self.active_target > self.provider_ceiling:
            raise ValueError('require 0 <= reserve < active_target <= provider_ceiling')
        if self.output_tokens+self.safety_margin >= self.provider_ceiling:
            raise ValueError('output allowance and safety margin leave no prompt capacity')

    @property
    def normal_threshold(self):
        return self.active_target-self.reserve

    @property
    def available_prompt_tokens(self):
        return self.provider_ceiling-self.output_tokens-self.safety_margin


def effective_request_body(request: dict) -> dict:
    """The OpenAI SDK merges extra_body over typed body fields before sending.

    Transport options are not template inputs. Query overrides are unsupported
    because they could alter server preprocessing outside the counted body.
    """
    if type(request) is not dict:
        raise ValueError('final request must be an exact dict')
    if request.get('extra_query'):
        raise ValueError('query overrides are not certified prompt inputs')
    body={k:v for k,v in request.items() if k not in ('extra_body','extra_headers','extra_query','timeout')}
    extra=request.get('extra_body')
    if extra is not None:
        if type(extra) is not dict:
            raise ValueError('extra_body must be an exact dict or None')
        body.update(extra)
    return _json_value(body)


def prompt_fingerprint(request: dict, identity: ProviderIdentity) -> str:
    # Headers are included conservatively even when they only select affinity.
    # Their contents stay private; the public count carries only this digest.
    headers=_json_value(request.get('extra_headers'))
    return _digest({'identity':asdict(identity), 'body':effective_request_body(request), 'headers':headers})


@dataclass(frozen=True)
class PromptCount:
    token_ids: tuple[int, ...]
    exact: bool
    prompt_fingerprint: str
    identity: ProviderIdentity
    bos_adjustment: int = 0

    def __post_init__(self):
        if type(self.token_ids) is not tuple or any(type(token) is not int or token < 0 for token in self.token_ids):
            raise ValueError('token_ids must be an immutable tuple of nonnegative integers')
        if type(self.exact) is not bool:
            raise ValueError('exact must be a boolean')
        if type(self.bos_adjustment) is not int or self.bos_adjustment not in (0,1):
            raise ValueError('only the fixed conservative zero/one BOS convention is supported')
        if type(self.prompt_fingerprint) is not str or len(self.prompt_fingerprint)!=64:
            raise ValueError('missing prompt fingerprint')
        if not isinstance(self.identity,ProviderIdentity):
            raise ValueError('missing provider identity')

    @property
    def tokens(self):
        return len(self.token_ids)+self.bos_adjustment


@dataclass(frozen=True)
class BudgetDecision:
    allowed: bool
    code: str
    budget_class: str
    prompt_tokens: int
    output_tokens: int
    safety_margin_tokens: int
    required_tokens: int
    available_tokens: int
    reason: str | None
    return_plan: str | None
    message: str
    prompt_fingerprint: str

    def to_dict(self):
        return asdict(self)


def enforce_final_request(request: dict, limits: BudgetLimits, *, count: PromptCount,
                          reason: str | None = None, return_plan: str | None = None) -> BudgetDecision:
    """Classify a complete counted payload without mutating it or its allowance."""
    prompt=count.tokens
    def result(allowed,code,classification='rejected'):
        return BudgetDecision(allowed,code,classification,prompt,limits.output_tokens,limits.safety_margin,
            prompt,limits.available_prompt_tokens,reason,return_plan,code.replace('_',' '),count.prompt_fingerprint)
    if not count.exact:
        return result(False,'exact_count_required')
    if count.identity.context_tokens != limits.provider_ceiling:
        return result(False,'provider_identity_changed')
    body=effective_request_body(request)
    if body.get('model') != count.identity.model:
        return result(False,'provider_identity_changed')
    if prompt_fingerprint(request,count.identity) != count.prompt_fingerprint:
        return result(False,'stale_prompt_count')
    caps=[body[k] for k in ('max_tokens','max_completion_tokens','n_predict','max_output_tokens') if k in body]
    if not caps or any(type(cap) is not int or cap != limits.output_tokens for cap in caps):
        return result(False,'output_allowance_changed')
    if prompt > limits.available_prompt_tokens:
        return result(False,'provider_window_exceeded')
    if prompt <= limits.normal_threshold:
        return result(True,'within_normal_budget','normal')
    if prompt <= limits.active_target:
        if reason not in ('restore','pending_group'):
            return result(False,'reserve_reason_required')
        return result(True,'reserve_used','reserve')
    if reason not in ('protected_group','management','restore'):
        return result(False,'unclassified_headroom')
    if type(return_plan) is not str or not return_plan.strip() or len(return_plan)>2000:
        return result(False,'return_plan_required')
    return result(True,'temporary_headroom','headroom')


@dataclass
class PressureEpisode:
    """Runtime state; the owning engine includes it in operation rollback."""
    attempts: int = 0
    nonprogress: int = 0
    max_attempts: int = 6
    max_nonprogress: int = 2
    reason: str | None = None
    return_plan: str | None = None
    # Monotonic pressure-episode id, bumped by every reset (recovery below the
    # normal threshold / new bind). The bounded recovery turn is offered at
    # most once per epoch.
    epoch: int = 0

    def __post_init__(self):
        for name in ('attempts','nonprogress'):
            _integer(getattr(self,name),name)
        for name in ('max_attempts','max_nonprogress'):
            _integer(getattr(self,name),name,minimum=1)
        if self.max_attempts>6 or self.max_nonprogress>2:
            raise ValueError('pressure limits cannot exceed six attempts or two nonprogress edits')

    def reset(self):
        self.attempts=self.nonprogress=0
        self.reason=self.return_plan=None
        self.epoch+=1

    def admit(self, prompt_tokens, reason, return_plan, limits):
        _integer(prompt_tokens,'prompt_tokens')
        if prompt_tokens <= limits.normal_threshold:
            self.reset()
            return True
        if prompt_tokens > limits.available_prompt_tokens:
            return False
        if prompt_tokens <= limits.active_target:
            if reason not in ('restore','pending_group'):
                return False
        elif reason not in ('protected_group','management','restore') or type(return_plan) is not str or not return_plan.strip():
            return False
        if self.attempts>=self.max_attempts or self.nonprogress>=self.max_nonprogress:
            return False
        self.attempts+=1
        self.reason,self.return_plan=reason,return_plan
        return True

    def observe_edit(self, before, after, normal_threshold):
        for name,value in (('before',before),('after',after),('normal_threshold',normal_threshold)):
            _integer(value,name)
        if after <= normal_threshold:
            self.reset()
        elif after >= before:
            self.nonprogress+=1
        else:
            self.nonprogress=0
