"""Shared fixtures for the hermes-clm test suite.

Every test runs against a throwaway state root so no test ever touches the
real ``~/.hermes/clm`` and canonical-history invariants are asserted, not
assumed.
"""

import sys
import os
import tempfile
import shutil

import pytest

# Make the package importable regardless of install state.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from hermes_clm import CLMConfig, CLMEngine  # noqa: E402
from hermes_clm.tools import dispatch        # noqa: E402


def _make_config(tmp_path, **overrides):
    cfg = CLMConfig()
    cfg.state_root = os.path.join(str(tmp_path), "clm")
    for k, v in overrides.items():
        setattr(cfg, k, v)
    cfg._validate()
    return cfg


@pytest.fixture
def tmp_home(tmp_path):
    return str(tmp_path)


@pytest.fixture
def config(tmp_path):
    return _make_config(tmp_path)


@pytest.fixture
def engine(config, tmp_path):
    eng = CLMEngine(config=config, hermes_home=str(tmp_path))
    eng.on_session_start("sess-test", conversation_id="conv-test")
    return eng


def _conversation(n_pairs=6, pad=120):
    """A deterministic system + n_pairs user/assistant turns."""
    msgs = [{"role": "system", "content": "You are a helpful assistant."}]
    for i in range(n_pairs):
        msgs.append({"role": "user",
                     "content": f"user turn {i} " + "u" * pad})
        msgs.append({"role": "assistant",
                     "content": f"assistant turn {i} " + "a" * pad})
    return msgs


def _call(engine, name, args):
    import json
    return json.loads(dispatch(engine, name, args))


def _compactable_uids(engine, limit=None):
    """UIDs of non-protected assistant messages (safe compact targets)."""
    out = [r["uid"] for r in engine.list_active()
           if r["role"] == "assistant" and not r["protected"]]
    return out[:limit] if limit else out


@pytest.fixture
def seeded_engine(engine):
    """An engine with a real conversation already ingested."""
    engine.on_turn_complete(_conversation())
    return engine


def cleanup(path):
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass
