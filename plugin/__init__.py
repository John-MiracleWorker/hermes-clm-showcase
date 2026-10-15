"""Hermes CLM plugin — Context Language Models for Hermes Agent.

Model-directed, reversible working-context management as a Hermes
ContextEngine. The canonical conversation stays untouched; the model
compacts, archives, and restores its own active context through the
clm_* tools.

Inspired by Context Language Models (an independent implementation;
no reference code is bundled).

Registration contract (Hermes plugin_loader / plugins/context_engine):
  - ``register(ctx)`` is invoked at agent init (per session, per profile);
    the engine is handed to the host via ``ctx.register_context_engine(engine)``.
  - The five clm_* tools are exposed through the engine's native
    ``get_tool_schemas()`` / ``handle_tool_call()`` path — the host's
    context-engine dispatch — so current-turn ingest is never lost.
  - Select with ``context.engine: hermes-clm`` in config.yaml. A
    user-installed engine needs no ``plugins.enabled`` entry. The
    profile's ``clm:`` config section (and ``context.clm:`` fallback) is
    read here and passed to ``CLMConfig.from_hermes`` so per-profile
    knobs like ``clm.recovery_turn`` reach the engine.
  - The model/provider context must still advertise >= 64K tokens (Hermes
    hard floor); the 32K CLM active target is an internal working-set
    budget and is never reported as the model context length.
"""

import logging
import os
import sys

logger = logging.getLogger(__name__)

# The hermes_clm package is expected to be importable (pip-installed, or on
# sys.path). Prefer a normal import; fall back to a checkout named
# `hermes-clm` in the home directory.
_REPO_PATHS = (
    os.path.expanduser("~/hermes-clm"),
)


def _import_engine():
    try:
        from hermes_clm import CLMConfig, CLMEngine  # noqa: F401
        import hermes_clm
        return CLMConfig, CLMEngine, hermes_clm.__file__
    except ImportError:
        for path in _REPO_PATHS:
            if path not in sys.path and os.path.isdir(os.path.join(path, "hermes_clm")):
                sys.path.insert(0, path)
        from hermes_clm import CLMConfig, CLMEngine  # noqa: F401
        import hermes_clm
        return CLMConfig, CLMEngine, hermes_clm.__file__


def _hermes_home() -> str:
    try:
        from hermes_cli.config import get_hermes_home
        return str(get_hermes_home())
    except Exception:
        return os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))


_PROFILE_CONFIG_NOTE = None


def _profile_config():
    """Parse the active profile's ``config.yaml`` (None when unavailable).

    ``register()`` runs per agent init with the profile's home active, but the
    host hands the engine no agent config; read it here so ``clm:`` settings
    are honored instead of silently defaulting.
    """
    global _PROFILE_CONFIG_NOTE
    path = os.path.join(_hermes_home(), "config.yaml")
    # Prefer the host's own loader — the CLI runs on a bundled interpreter
    # without PyYAML, so a bare ``import yaml`` here would silently fail.
    # ``load_config_readonly`` resolves the same active home as _hermes_home().
    try:
        from hermes_cli.config import load_config_readonly
        data = load_config_readonly() or {}
        if isinstance(data, dict) and data:
            _PROFILE_CONFIG_NOTE = (f"host-loader ok keys={len(data)} has_clm={'clm' in data} "
                                    f"clm={str(data.get('clm'))[:80]}")
            return dict(data)
        _PROFILE_CONFIG_NOTE = f"host-loader empty ({type(data).__name__})"
    except Exception as exc:
        _PROFILE_CONFIG_NOTE = f"host-loader failed: {type(exc).__name__}: {exc}"
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if isinstance(data, dict):
            _PROFILE_CONFIG_NOTE += f"; yaml ok has_clm={'clm' in data}"
            return data
        _PROFILE_CONFIG_NOTE += f"; yaml not-a-dict type={type(data).__name__}"
        return None
    except Exception as exc:
        _PROFILE_CONFIG_NOTE += f"; yaml {type(exc).__name__}: {exc} path={path}"
        return None


# The host may call register() more than once for the same engine (discovery
# probe, then the real load). The command handler must always target the
# engine the host actually uses — i.e. the last one constructed here.
_ACTIVE_ENGINE = None


def get_active_engine():
    """The CLMEngine instance most recently registered with the host."""
    return _ACTIVE_ENGINE


# --- diagnostics --------------------------------------------------------

_TRACE_PATH = os.environ.get("HERMES_CLM_TRACE_FILE")


def _trace_line(text):
    if not _TRACE_PATH:
        return
    try:
        with open(_TRACE_PATH, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")
    except Exception:
        pass


def _trace_wrap_engine_class(cls):
    """Trace host→engine calls (get_tool_schemas / on_session_start) once per process."""
    if getattr(cls, "_hermes_clm_trace_wrapped", False):
        return
    cls._hermes_clm_trace_wrapped = True
    for meth_name in ("get_tool_schemas", "on_session_start"):
        orig = getattr(cls, meth_name, None)
        if not callable(orig):
            continue

        def make(orig=orig, meth_name=meth_name):
            def _wrapped(self, *args, **kwargs):
                try:
                    result = orig(self, *args, **kwargs)
                except BaseException as exc:
                    _trace_line(f"{meth_name} RAISED {type(exc).__name__}: {exc}")
                    raise
                if meth_name == "get_tool_schemas":
                    names = []
                    for s in (result or []):
                        inner = s.get("function", s) if isinstance(s, dict) else {}
                        names.append(inner.get("name", "?"))
                    _trace_line(f"get_tool_schemas -> {len(result or [])} {names}")
                else:
                    _trace_line(f"{meth_name} called (nargs={len(args)})")
                return result
            return _wrapped

        setattr(cls, meth_name, make())


def register(ctx):
    """Plugin entry point — register the CLM context engine."""
    global _ACTIVE_ENGINE
    CLMConfig, CLMEngine, engine_pkg_file = _import_engine()
    config = CLMConfig.from_hermes(_profile_config(), _hermes_home())
    engine = CLMEngine(config=config, hermes_home=_hermes_home())
    _ACTIVE_ENGINE = engine

    _trace = os.environ.get("HERMES_CLM_TRACE_FILE")
    if _trace:
        try:
            with open(_trace, "a", encoding="utf-8") as _fh:
                _fh.write(f"{__file__} home={_hermes_home()} pkg={engine_pkg_file} "
                          f"recovery_turn={config.recovery_turn} pid={os.getpid()}\n"
                          f"  cfg_note={_PROFILE_CONFIG_NOTE}\n"
                          f"  python={sys.executable} {sys.version.split()[0]}\n")
        except Exception:
            pass

    ctx.register_context_engine(engine)

    _trace_wrap_engine_class(type(engine))

    # Optional operator inspection (spec §48/§49): ``/clm`` in chat.
    register_command = getattr(ctx, "register_command", None)
    if callable(register_command):
        # The loader may import this file as a standalone module (synthetic
        # package name with a hyphen), so load the sibling by file path.
        import importlib.util
        _path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "clm_command.py")
        _spec = importlib.util.spec_from_file_location(
            "_hermes_clm_command", _path)
        clm_command = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(clm_command)
        register_command(
            "clm",
            lambda raw_args: clm_command.handle_clm_command(raw_args, None),
            description="Inspect CLM working-context state (status/list/restore/events)",
            args_hint="[status|active|archive|summary <id>|events]",
        )

    logger.info(
        "hermes-clm registered: home=%s engine_pkg=%s active_target=%d reserve=%d "
        "provider_min=%d recovery_turn=%s budget_mode=%s",
        _hermes_home(), engine_pkg_file, config.active_target_tokens,
        config.reserve_tokens, config.provider_context_min_tokens,
        config.recovery_turn, config.budget_mode,
    )
