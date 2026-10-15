"""Repo-root pytest conftest: make the Hermes host importable for tests.

The engine subclasses the host ``ContextEngine`` when available. Tests run in
the project venv (system-site-packages off for host code), so expose the
Hermes checkout path explicitly.
"""

import os
import sys

_HOST = os.environ.get(
    "HERMES_AGENT_PATH", os.path.expanduser("~/.hermes/hermes-agent"))
if _HOST not in sys.path and os.path.isdir(os.path.join(_HOST, "agent")):
    sys.path.insert(0, _HOST)
