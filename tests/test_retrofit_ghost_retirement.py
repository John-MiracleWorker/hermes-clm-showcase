"""Retrofit adoption for pre-map stores (clm.retrofit_retire_ghosts).

A legacy store whose history was replaced while the map was off keeps
branch-external superseded versions ("ghosts") that can never be witnessed or
audited, so coverage refuses forever. With the flag enabled, strictly
superseded ghosts are retired on load and byte-preserved in a sidecar next to
state.json; everything else keeps the ordinary provenance refusal.
All-or-nothing and fail-closed.
"""
from copy import deepcopy
import json
import os
from pathlib import Path

from hermes_clm.store import ContextStore
from conftest import _make_config
from test_context_map import engine_for, originals


def _superseded_ghost_state(tmp_path, conversation):
    """Second ingest drops a stale extra envelope key from the tool result —
    an envelope-normalization change: the retired version carries `name`, its
    successor does not. The state's ownership identity is this conversation's,
    so it loads back under the same id. Returns (state, ghost_uid, successor_uid)."""
    eng = engine_for(tmp_path / "old", enabled=False, conversation=conversation)
    stripped = deepcopy(originals())
    stripped[3] = {k: v for k, v in stripped[3].items() if k != "name"}
    eng.store.ingest_messages(stripped)
    state = eng.store.state_dict()
    order = set(state["order"])
    same_slot = [u for u, r in state["messages"].items() if r["index"] == 3]
    ghost = next(u for u in same_slot if u not in order)
    successor = next(u for u in same_slot if u in order)
    return state, ghost, successor


def _adopt_cfg(tmp_path, *, retire):
    cfg = _make_config(tmp_path, protect_latest_user_turns=0,
                       protect_latest_assistant_turns=0)
    cfg.context_map_enabled = True
    cfg.retrofit_retire_ghosts = retire
    return cfg


def _stage(tmp_path, state, conversation):
    cfg = _adopt_cfg(tmp_path, retire=True)
    base = Path(cfg.state_root) / conversation
    base.mkdir(parents=True, exist_ok=True)
    (base / "state.json").write_text(json.dumps(state))
    return cfg, base


def test_superseded_ghost_is_retired_on_load(tmp_path):
    state, ghost, successor = _superseded_ghost_state(tmp_path, "retrofit-adopt")
    cfg, base = _stage(tmp_path, state, "retrofit-adopt")
    store = ContextStore.load(cfg.state_root, "retrofit-adopt", cfg)
    assert store.recovery_required is None
    assert ghost not in store.messages
    assert successor in store.messages
    sidecars = list(base.glob("retired-legacy-ghosts-*.json"))
    assert len(sidecars) == 1
    payload = json.loads(sidecars[0].read_text())
    assert set(payload["retired"]) == {ghost}
    assert payload["retired"][ghost]["envelope"] == state["messages"][ghost]["envelope"]
    # The adoption receipt is published transactionally with the next save.
    store.save(cfg.state_root)
    events = [json.loads(line) for line in (base / "events.jsonl").read_text().splitlines()]
    kinds = [e.get("event") for e in events]
    assert "legacy_ghosts_retired" in kinds
    assert "context_map_backfill" in kinds
    receipt = next(e for e in events if e.get("event") == "legacy_ghosts_retired")
    assert receipt["initiator"] == "host-retrofit"
    assert receipt["retired"] == [ghost]
    assert receipt["sidecar"].endswith(".json")
    again = ContextStore.load(cfg.state_root, "retrofit-adopt", cfg)
    assert again.recovery_required is None
    assert ghost not in again.messages
    assert "context_map" in again.state_dict()


def test_flag_off_keeps_the_refusal(tmp_path):
    state, ghost, successor = _superseded_ghost_state(tmp_path, "retrofit-off")
    cfg = _adopt_cfg(tmp_path, retire=False)
    base = Path(cfg.state_root) / "retrofit-off"
    base.mkdir(parents=True, exist_ok=True)
    (base / "state.json").write_text(json.dumps(state))
    store = ContextStore.load(cfg.state_root, "retrofit-off", cfg)
    assert store.recovery_required == "context_map_provenance_unverifiable"
    assert store.legacy_state is not None
    assert ghost in store.legacy_state["messages"]
    assert not list(base.glob("retired-legacy-ghosts-*.json"))


def test_content_divergence_refuses_even_with_flag(tmp_path):
    eng = engine_for(tmp_path / "old", enabled=False, conversation="retrofit-diverged")
    changed = originals()
    changed[2]["tool_calls"][0]["function"]["arguments"] = '{"path":"corrected-report"}'
    eng.store.ingest_messages(changed)
    state = eng.store.state_dict()
    cfg, base = _stage(tmp_path, state, "retrofit-diverged")
    store = ContextStore.load(cfg.state_root, "retrofit-diverged", cfg)
    assert store.recovery_required == "context_map_provenance_unverifiable"
    assert not list(base.glob("retired-legacy-ghosts-*.json"))


def test_retirement_is_all_or_nothing(tmp_path):
    eng = engine_for(tmp_path / "old", enabled=False, conversation="retrofit-mixed")
    stripped = deepcopy(originals())
    stripped[3] = {k: v for k, v in stripped[3].items() if k != "name"}
    eng.store.ingest_messages(stripped)
    changed = deepcopy(stripped)
    changed[2]["tool_calls"][0]["function"]["arguments"] = '{"path":"corrected-report"}'
    eng.store.ingest_messages(changed)
    state = eng.store.state_dict()
    order = set(state["order"])
    ghosts = {u for u, r in state["messages"].items()
              if r.get("archived") and u not in order}
    assert len(ghosts) == 2
    cfg, base = _stage(tmp_path, state, "retrofit-mixed")
    store = ContextStore.load(cfg.state_root, "retrofit-mixed", cfg)
    assert store.recovery_required == "context_map_provenance_unverifiable"
    assert store.legacy_state is not None
    assert ghosts.issubset(set(store.legacy_state["messages"]))
    assert not list(base.glob("retired-legacy-ghosts-*.json"))


def test_sidecar_write_failure_is_fail_closed(tmp_path):
    state, ghost, successor = _superseded_ghost_state(tmp_path, "retrofit-readonly")
    cfg, base = _stage(tmp_path, state, "retrofit-readonly")
    os.chmod(base, 0o555)
    try:
        store = ContextStore.load(cfg.state_root, "retrofit-readonly", cfg)
    finally:
        os.chmod(base, 0o755)
    assert store.recovery_required == "context_map_provenance_unverifiable"
    assert not list(base.glob("retired-legacy-ghosts-*.json"))


def test_healthy_legacy_load_is_untouched(tmp_path):
    eng = engine_for(tmp_path / "old", enabled=False, conversation="retrofit-clean")
    state = eng.store.state_dict()
    cfg, base = _stage(tmp_path, state, "retrofit-clean")
    store = ContextStore.load(cfg.state_root, "retrofit-clean", cfg)
    assert store.recovery_required is None
    assert len(store.messages) == len(state["messages"])
    assert not list(base.glob("retired-legacy-ghosts-*.json"))


def test_witnessed_map_ghost_and_sqlite_tool_reload_remain_healthy(tmp_path):
    from hermes_clm import CLMEngine
    cfg = _adopt_cfg(tmp_path, retire=True)
    eng = CLMEngine(cfg, str(tmp_path))
    eng.on_session_start("oneshot-reload", conversation_id="oneshot-reload")
    live = deepcopy(originals())
    live[3]["tool_name"] = live[3]["name"]
    live[3]["_db_persisted"] = True
    eng.select_context(live, conversation_messages=live, incoming_message=live[-1])
    eng.on_turn_complete(live[1:])

    # Next --oneshot CLI turn reloads from SQLite without the "name" key on role="tool".
    reloaded = deepcopy(live)
    reloaded[3].pop("name", None)
    reloaded.append({"role": "user", "content": "turn 11 follow-up"})
    eng2 = CLMEngine(cfg, str(tmp_path))
    eng2.on_session_start("oneshot-reload", conversation_id="oneshot-reload")
    assert eng2.store.recovery_required is None
    selected = eng2.select_context(reloaded, conversation_messages=reloaded, incoming_message=reloaded[-1])
    assert selected is not None
    assert eng2.store.recovery_required is None
    order_set = set(eng2.store.order)
    ghosts = [u for u in eng2.store.messages if u not in order_set]
    assert ghosts == []

    # Even if a map-enabled state already contains a witnessed ghost on disk,
    # retrofit_retire_ghosts must not delete the witnessed ghost and corrupt validate_map.
    eng.store.ingest_messages(reloaded[:-1])
    eng._persist()
    reloaded_store = ContextStore.load(cfg.state_root, "oneshot-reload", cfg)
    assert reloaded_store.recovery_required is None
