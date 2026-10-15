# Hermes plugin shim

Thin plugin that registers the CLM context engine with Hermes Agent
(`register(ctx)` → `ctx.register_context_engine(engine)`), and ships the `/clm`
inspection command.

## Install

1. Make the `hermes_clm` package importable — from the repo root:

   ```bash
   pip install -e .
   ```

   (Or keep a checkout named `hermes-clm` in your home directory; the shim
   falls back to it when no installed package is found.)

2. Copy this directory into the plugins dir of every profile that should use CLM:

   - default profile: `~/.hermes/plugins/hermes-clm/`
   - profile `<p>`:   `~/.hermes/profiles/<p>/plugins/hermes-clm/`

3. Select the engine in the profile's `config.yaml`:

   ```yaml
   context:
     engine: hermes-clm
   ```

4. The provider context must advertise **≥ 64K tokens** (a Hermes floor, not a
   CLM setting). The CLM active target (~32K by default) is an internal
   working-set budget and is never reported as the model context length.

Per-profile knobs live under a `clm:` section (e.g. `clm.recovery_turn`,
`clm.context_map_enabled`, `clm.active_target_tokens`); the shim reads them
through the host's own config loader and passes them to `CLMConfig.from_hermes`
— without that wiring, knobs silently stay at defaults.

## Notes for plugin authors (learned the hard way)

- The `hermes` CLI runs on a bundled interpreter that may not have PyYAML —
  never `import yaml` in a plugin; read config through
  `hermes_cli.config.load_config_readonly`.
- The host can call `register()` more than once per process (discovery probe,
  then the real load); keep a module-level handle to the last engine built.
- Set `HERMES_CLM_TRACE_FILE=/some/file` to append a diagnostic line on every
  plugin load (shim path, resolved home, package path, interpreter).
