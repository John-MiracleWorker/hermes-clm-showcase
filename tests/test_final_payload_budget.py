"""Exact final-prompt math, full-body identity, and bounded pressure policy.

Synthetic token IDs exercise arithmetic/control flow only, never Qwen parity.
"""
import importlib
from dataclasses import replace
import pytest

def api():
    return importlib.import_module('hermes_clm.provider_budget')

def identity():
    return api().ProviderIdentity(provider='llama.cpp', base_url='https://invalid.example/v1', model='offline',
        model_sha256='1'*64, tokenizer_sha256='2'*64, template_sha256='3'*64,
        server_sha256='4'*64, build_id='offline-build', instance_id='offline-process',
        preprocessing_sha256='5'*64, context_tokens=131072)

def count(n,request=None,exact=True):
    b=api(); request=request or {'model':'offline','messages':[{'role':'user','content':'x'}],'max_tokens':4096}
    return b.PromptCount(tuple(range(n)),exact,b.prompt_fingerprint(request,identity()),identity())

@pytest.mark.parametrize('prompt,reason,plan,allowed,classification',[
    (28671,None,None,True,'normal'),(28672,None,None,True,'normal'),
    (28673,None,None,False,'rejected'),(28673,'pending_group',None,True,'reserve'),
    (32767,'restore',None,True,'reserve'),(32768,'restore',None,True,'reserve'),
    (32769,None,None,False,'rejected'),(32769,'management',None,False,'rejected'),
    (32769,'management','Archive prior completed sources to return below S',True,'headroom'),
    (125951,'protected_group','Complete pending group, then archive old evidence',True,'headroom'),
    (125952,'restore','Use restored source, then archive it to return below S',True,'headroom'),
    (125953,'restore','Return below S',False,'rejected')])
def test_complete_payload_boundaries(prompt,reason,plan,allowed,classification):
    b=api();request={'model':'offline','messages':[{'role':'user','content':'x'}],'max_tokens':4096}
    decision=b.enforce_final_request(request,b.BudgetLimits(),count=count(prompt,request),reason=reason,return_plan=plan)
    assert decision.allowed is allowed
    assert decision.budget_class==classification
    assert decision.prompt_tokens==prompt and decision.output_tokens==4096
    assert decision.available_tokens==125952
    assert decision.required_tokens==prompt

@pytest.mark.parametrize('variant', ['messages','tools','thinking','model','template','extra_body','ceiling','build','instance'])
def test_full_request_or_identity_change_invalidates_fingerprint(variant):
    b=api();request={'model':'offline','messages':[{'role':'user','content':'x'}],'max_tokens':4096}; ident=identity()
    old=b.prompt_fingerprint(request,ident)
    if variant=='ceiling':ident=replace(ident,context_tokens=65536)
    elif variant=='build':ident=replace(ident,build_id='other-build')
    elif variant=='instance':ident=replace(ident,instance_id='other-process')
    else:request[variant]={'changed':'中 🚑 code()'}
    assert b.prompt_fingerprint(request,ident)!=old

@pytest.mark.parametrize('change,code', [('approximate','exact_count_required'),('stale','stale_prompt_count'),('ceiling','provider_identity_changed'),('output','output_allowance_changed')])
def test_no_approximate_stale_or_silent_output_reduction(change,code):
    b=api();request={'model':'offline','messages':[{'role':'user','content':'x'}],'max_tokens':4096}; c=count(100,request)
    limits=b.BudgetLimits()
    if change=='approximate':c=replace(c,exact=False)
    elif change=='stale':request['messages'].append({'role':'assistant','content':'new'})
    elif change=='ceiling':limits=replace(limits,provider_ceiling=65536)
    else:request['max_tokens']=2048;c=count(100,request)
    d=b.enforce_final_request(request,limits,count=c)
    assert not d.allowed and d.code==code


def test_extra_body_is_actual_wire_authority_and_preserves_input():
    b=api();messages=[{'role':'user','content':None}]
    request={'model':'offline','messages':[], 'tools':[], 'max_tokens':4096,
             'extra_body':{'messages':messages,'tools':[{'type':'function','function':{'name':'x','parameters':{'type':'object'}}}],'chat_template_kwargs':{'enable_thinking':True}}, 'timeout':30}
    body=b.effective_request_body(request)
    assert body['messages']==messages and body['tools'] and body['chat_template_kwargs']['enable_thinking']
    assert 'timeout' not in body and 'extra_body' not in body
    body['messages'][0]['content']='changed'
    assert messages[0]['content'] is None and request['messages']==[]

@pytest.mark.parametrize('bad',[True,-1,1.5,'4096'])
def test_invalid_limits_reject_before_budget_arithmetic(bad):
    with pytest.raises(ValueError):api().BudgetLimits(output_tokens=bad)


def test_pressure_attempts_and_two_nonprogress_edits_are_bounded():
    b=api();episode=b.PressureEpisode()
    for _ in range(6):assert episode.admit(34000,'management','Return below S',b.BudgetLimits())
    assert not episode.admit(34000,'management','Return below S',b.BudgetLimits())
    episode=b.PressureEpisode()
    assert episode.admit(34000,'management','Return below S',b.BudgetLimits())
    episode.observe_edit(34000,34000,28672)
    assert episode.admit(34000,'management','Return below S',b.BudgetLimits())
    episode.observe_edit(34000,35000,28672)
    assert not episode.admit(35000,'management','Return below S',b.BudgetLimits())
    episode.observe_edit(35000,28000,28672)
    assert episode.attempts==episode.nonprogress==0


def test_provider_output_override_cannot_exceed_reserved_allowance():
    b=api();request={'model':'offline','messages':[{'role':'user','content':'x'}],
                     'max_tokens':4096,'extra_body':{'n_predict':131072}}
    d=b.enforce_final_request(request,b.BudgetLimits(),count=count(100,request))
    assert not d.allowed and d.code=='output_allowance_changed'
