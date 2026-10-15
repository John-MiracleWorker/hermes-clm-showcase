"""Persistence, pressure and restore limits at the atomic engine boundary."""
import json
from copy import deepcopy
import pytest
from hermes_clm.config import CLMConfig
from hermes_clm.engine import CLMEngine
from hermes_clm.provider_budget import ProviderIdentity, PromptCount, prompt_fingerprint
from hermes_clm.policies import PolicyViolation


class Counter:
    tokens=100
    def count_final_request(self,request,identity):
        return PromptCount((1,)*self.tokens,True,prompt_fingerprint(request,identity),identity)


def setup(tmp_path):
    cfg=CLMConfig(persistence=True,event_log=True,state_root=str(tmp_path),budget_mode='exact',
                  protect_latest_user_turns=0,protect_latest_assistant_turns=0)
    e=CLMEngine(cfg);e.on_session_start('atomic')
    ident=ProviderIdentity('custom','https://invalid.example/v1','offline','1'*64,'2'*64,
                           '3'*64,'4'*64,'fixture-build','fixture-process','5'*64,131072)
    counter=Counter();e.bind_exact_counter(counter,ident)
    history=[{'role':'system','content':'instructions'},{'role':'user','content':'root'},
        {'role':'assistant','content':'old factual source material'},
        {'role':'user','content':'new task'}]
    messages=e.select_context(history,conversation_messages=history)
    route={'provider':'custom','base_url':ident.base_url,'api_mode':'chat_completions'}
    assert e.check_final_request({'model':'offline','messages':messages,'max_tokens':4096},route=route)['allowed']
    return e,counter,e.store.order[2]


def accepted(e,tmp_path):
    return (e.store.state_dict(),deepcopy(e._budget_frame),deepcopy(e._pressure_episode),
            {str(p.relative_to(tmp_path)):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()})


def test_persistence_failure_restores_exact_frame_and_pressure(tmp_path,monkeypatch):
    e,counter,uid=setup(tmp_path);before=accepted(e,tmp_path)
    import hermes_clm.store as store
    def refuse(*args):raise OSError('injected atomic replace failure')
    monkeypatch.setattr(store.os,'replace',refuse)
    with pytest.raises(OSError,match='injected'):e.apply_compact([uid],'small')
    assert accepted(e,tmp_path)==before


def test_two_real_nonprogressing_edits_stop_headroom_send(tmp_path):
    e,counter,uid=setup(tmp_path);summary=e.apply_compact([uid],'small')['summary_id']
    counter.tokens=35000;e.begin_budget_excursion('management','Compact stale material below 28672.')
    assert e.check_final_request(e._budget_frame['request'],route=e._budget_frame['route'])['allowed']
    e.apply_replace_summary(summary,'still same size')
    assert e._pressure_episode.nonprogress==1
    e.apply_replace_summary(summary,'still not smaller')
    assert e._pressure_episode.nonprogress==2
    result=e.check_final_request(e._budget_frame['request'],route=e._budget_frame['route'])
    assert not result['allowed'] and result['code']=='pressure_episode_exhausted'


def test_exact_restore_cap_uses_rendered_delta_not_legacy_estimate(tmp_path):
    e,counter,uid=setup(tmp_path);e.apply_compact([uid],'small')
    e.config.max_restoration_tokens=1
    # This offline exact oracle reports zero growth, despite positive store tokens.
    assert e.store.messages[uid].token_count>1
    assert e.apply_restore([uid])['ok']


def test_exact_restore_cap_rejects_rendered_growth_and_rolls_back(tmp_path):
    e,counter,uid=setup(tmp_path);e.apply_compact([uid],'small')
    e.config.max_restoration_tokens=100;counter.tokens=201
    before=accepted(e,tmp_path)
    with pytest.raises(PolicyViolation) as failure:e.apply_restore([uid])
    assert failure.value.code=='max_restoration_exceeded'
    assert failure.value.required_tokens==101 and failure.value.available_tokens==100
    assert accepted(e,tmp_path)==before


def test_store_transaction_does_not_exempt_direct_writer_from_budget(tmp_path):
    e,counter,uid=setup(tmp_path);summary=e.apply_compact([uid],'small')['summary_id']
    e.store.begin_transaction();before=accepted(e,tmp_path);counter.tokens=130000
    with pytest.raises(PolicyViolation):e.store.update_summary(summary,'oversized')
    assert accepted(e,tmp_path)==before


def test_reserve_restore_is_classified_on_next_actual_request(tmp_path):
    e,counter,uid=setup(tmp_path);e.apply_compact([uid],'small');counter.tokens=30000
    assert e.apply_restore([uid])['ok']
    result=e.check_final_request(e._budget_frame['request'],route=e._budget_frame['route'])
    assert result['allowed'] and result['budget_class']=='reserve' and result['reason']=='restore'
