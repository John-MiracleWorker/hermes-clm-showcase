"""Lossless request views: normalization, aliasing and role collapse regressions."""
from copy import deepcopy

from hermes_clm import CLMEngine
from hermes_clm.store import ContextStore
from conftest import _make_config


def envelopes():
    return [
        {"role": "system", "content": [{"type": "text", "text": "first"}]},
        {"role": "developer", "content": "second", "name": "policy"},
        {"role": "system", "content": "third"},
        {"role": "user", "content": [{"type": "text", "text": "task"},
         {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA", "detail": "low"}}]},
        {"role": "assistant", "content": None, "reasoning_content": "thinking",
         "reasoning_details": [{"type": "reasoning.encrypted", "data": "opaque", "signature": "sig"}],
         "tool_calls": [{"id": "exact", "type": "function", "function": {
             "name": "read", "arguments": ' { "path" : "a" }\n'}}]},
        {"role": "tool", "content": "", "tool_call_id": "exact", "name": "read"},
        {"role": "assistant", "content": "", "refusal": None,
         "audio": {"id": "audio", "data": "opaque"}, "provider_extension": {"nested": [1, None]}},
    ]


def originals_in(selected, originals):
    # Guidance is synthetic; every original must be byte/type-equivalent and ordered.
    assert selected is not None
    retained = [m for m in selected if m in originals]
    assert retained == originals


def test_envelope_ingest_select_save_load(tmp_path):
    cfg = _make_config(tmp_path, emit_nudges=False, tool_output_soft_limit_chars=0)
    eng = CLMEngine(cfg, str(tmp_path))
    eng.on_session_start("envelopes")
    source = envelopes()
    before = deepcopy(source)
    originals_in(eng.select_context(source), before)
    assert source == before
    eng.store.save(cfg.state_root)
    replay = CLMEngine(cfg, str(tmp_path))
    replay.on_session_start("envelopes")
    originals_in(replay.select_context(source), before)
    assert source == before


def test_returned_nested_envelopes_are_detached(tmp_path):
    cfg = _make_config(tmp_path, emit_nudges=False)
    eng = CLMEngine(cfg, str(tmp_path))
    eng.on_session_start("immutable")
    source = envelopes()
    selected = eng.select_context(source)
    call = next(m for m in selected if m.get("tool_calls"))
    call["tool_calls"][0]["function"]["arguments"] = "mutated"
    source[4]["reasoning_details"][0]["data"] = "input mutated"
    originals_in(eng.selector.build(), envelopes())
    record = next(r for r in eng.store.messages.values() if r.role == "assistant" and r.envelope.get("tool_calls"))
    record.envelope["tool_calls"][0]["id"] = "mutated"
    saved = eng.store.state_dict()
    saved["messages"][record.uid]["envelope"]["reasoning_details"][0]["data"] = "mutated"
    originals_in(eng.selector.build(), envelopes())


def test_explicit_envelope_adapter_strips_only_internal_fields():
    from hermes_clm.protocol import canonical_envelope, provider_envelope
    msg = {"role": "developer", "content": None, "name": "policy", "timestamp": 10,
           "message_uid": "host", "_row_id": 4, "_tool_call_uid": "occurrence",
           "display_metadata": {"private": True}, "uid": "internal", "protected": True,
           "_clm_future": "secret", "reasoning_details": [{"signature": "s"}],
           "provider_extension": {"cache_control": {"type": "ephemeral"}}}
    canonical = canonical_envelope(msg)
    assert canonical == msg
    wire = provider_envelope(canonical)
    assert wire == {"role": "developer", "content": None, "name": "policy",
                    "reasoning_details": [{"signature": "s"}],
                    "provider_extension": {"cache_control": {"type": "ephemeral"}}}
    wire["provider_extension"]["cache_control"]["type"] = "mutated"
    assert canonical == msg


def test_content_type_change_rewinds(tmp_path):
    store = ContextStore(_make_config(tmp_path))
    store.ingest_messages([{"role": "user", "content": None}, {"role": "assistant", "content": "tail"}])
    old = list(store.order)
    store.ingest_messages([{"role": "user", "content": ""}])
    assert store.order != old[:1]
    assert all(store.messages[u].archived and not store.messages[u].active for u in old)


def test_host_metadata_stripping_does_not_discard_context_edits(tmp_path):
    from hermes_clm.protocol import provider_envelope
    cfg = _make_config(tmp_path, emit_nudges=False, protect_latest_assistant_turns=0,
                       protect_latest_user_turns=0)
    eng = CLMEngine(cfg, str(tmp_path))
    eng.on_session_start("host-metadata")
    source = [{"role": "system", "content": "rules", "message_uid": "s"},
              {"role": "user", "content": "root", "message_uid": "u"},
              {"role": "assistant", "content": "old", "message_uid": "a"},
              {"role": "user", "content": "latest", "message_uid": "u2"}]
    eng.on_turn_complete(source)
    uid = eng.store._by_index[2]
    eng.apply_archive([uid])
    request = [provider_envelope(m) for m in source]
    selected = eng.select_context(request, conversation_messages=source, incoming_message=source[-1])
    assert uid == eng.store._by_index[2]
    assert eng.store.get(uid).archived
    assert request[2] not in selected
    assert all("message_uid" not in m for m in selected)


def test_full_host_request_single_message_is_a_rewind(tmp_path):
    cfg = _make_config(tmp_path, emit_nudges=False)
    eng = CLMEngine(cfg, str(tmp_path))
    eng.on_session_start("rewind")
    eng.on_turn_complete(envelopes())
    old = list(eng.store.order)
    now = [{"role": "user", "content": "new root"}]
    selected = eng.select_context(now, conversation_messages=now, incoming_message=now[0])
    assert selected == now
    assert all(not eng.store.get(u).active for u in old)


def test_runtime_instruction_replacement_keeps_existing_archives(tmp_path):
    cfg = _make_config(tmp_path, emit_nudges=False, protect_latest_assistant_turns=0,
                       protect_latest_user_turns=0)
    eng = CLMEngine(cfg, str(tmp_path))
    eng.on_session_start('instruction-refresh')
    source = envelopes()
    eng.on_turn_complete(source)
    call = eng.store._by_index[4]
    result = eng.store._by_index[5]
    summary = eng.apply_compact([result], 'historical read was empty')['summary_id']
    old_system = eng.store._by_index[0]
    changed = deepcopy(source)
    changed[0]['content'] = [{'type': 'text', 'text': 'refreshed runtime instruction'}]
    selected = eng.select_context(changed, conversation_messages=changed)
    assert changed[0] in selected
    assert source[1] in selected and source[2] in selected
    assert eng.store.get(call).archived and eng.store.get(result).archived
    assert eng.store.summary(summary).active
    assert eng.store.summary(summary).source_message_ids == [call, result]
    assert eng.store.get(old_system).archived and not eng.store.get(old_system).active
