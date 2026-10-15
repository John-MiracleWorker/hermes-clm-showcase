"""Model-directed compaction (spec §11, §21, §32.4, §33)."""

import pytest

from conftest import _call, _compactable_uids


def test_compact_archives_sources_and_creates_summary(seeded_engine):
    eng = seeded_engine
    targets = _compactable_uids(eng, 2)
    assert targets
    before = eng.store.working_set_tokens()
    rev_before = eng.store.revision

    res = _call(eng, "clm_compact", {"message_ids": targets,
                                     "summary": "Investigated X. Concluded Y."})
    assert res["ok"], res
    # Sources archived, not destroyed.
    for uid in targets:
        rec = eng.store.get(uid)
        assert rec is not None
        assert rec.archived
    # Summary created with provenance (spec §32.4).
    sid = res["summary_id"]
    s = eng.store.summary(sid)
    assert s is not None
    assert set(s.source_message_ids) == set(targets)
    # Revision advanced (spec §11 step 7).
    assert res["context_revision"] == rev_before + 1
    # Token accounting sane.
    assert res["tokens_before"] >= res["tokens_after"]
    assert res["tokens_saved"] > 0
    # The selected request now carries the summary, not the raw span.
    sel = eng.select_context([{"role": "user", "content": "next turn"}])
    joined = " ".join(str(m.get("content", "")) for m in sel)
    assert "Investigated X. Concluded Y." in joined


def test_compact_unknown_id_rejected(seeded_engine):
    res = _call(seeded_engine, "clm_compact",
                {"message_ids": ["msg_doesnotexist"], "summary": "x"})
    assert res["ok"] is False
    assert res.get("error") in ("unknown_message_ids", "unknown_ids")


def test_compact_empty_rejected(seeded_engine):
    res = _call(seeded_engine, "clm_compact",
                {"message_ids": [], "summary": "x"})
    assert res["ok"] is False


def test_compact_atomic_on_invalid(seeded_engine):
    # Mixing a valid and an invalid id must not partially apply (spec §33).
    eng = seeded_engine
    good = _compactable_uids(eng, 1)
    rev_before = eng.store.revision
    res = _call(eng, "clm_compact",
                {"message_ids": good + ["bogus_uid"], "summary": "x"})
    assert res["ok"] is False
    assert eng.store.revision == rev_before  # rolled back, no partial state
    for uid in good:
        assert not eng.store.get(uid).archived


def test_replace_summary_updates_in_place(seeded_engine):
    eng = seeded_engine
    targets = _compactable_uids(eng, 2)
    c = _call(eng, "clm_compact",
              {"message_ids": targets, "summary": "first version"})
    sid = c["summary_id"]
    res = _call(eng, "clm_replace_summary",
                {"summary_id": sid,
                 "replacement": "Refined: root cause was the driver."})
    assert res["ok"], res
    s = eng.store.summary(sid)
    assert "Refined" in s.content
    assert s.revision == 2
    # Provenance untouched.
    assert set(s.source_message_ids) == set(targets)


def test_replace_summary_unknown_rejected(seeded_engine):
    res = _call(seeded_engine, "clm_replace_summary",
                {"summary_id": "clm_summary_9999", "replacement": "x"})
    assert res["ok"] is False
