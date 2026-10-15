# Durable experiment smoke fixtures

These fixtures exercise the saved-experiment CLI without a model, provider, or
network call.  They are regression fixtures for configuration, subprocess
isolation, field mapping, checkpointing, and report comparison.  They are not
capability benchmarks and their passing score says nothing about an agent.

`smoke-ab.yaml` runs two equivalent single-process targets over the same two
transparent arithmetic records.  `multiagent-compatible.yaml` uses a target
that accepts a routed hand-off input, but it is still one deterministic
subprocess; it does not claim that multiple agents ran.

Run from the repository root with a fresh output directory:

```powershell
uv run --no-sync agentic-evalkit experiments validate examples/experiments/smoke-ab.yaml
uv run --no-sync agentic-evalkit experiments run examples/experiments/smoke-ab.yaml --run-dir C:/temp/evalkit-smoke-ab
uv run --no-sync agentic-evalkit experiments resume C:/temp/evalkit-smoke-ab
uv run --no-sync agentic-evalkit experiments compare C:/temp/evalkit-smoke-ab
```

The experiment records both arms as task passes because the fixture target is
programmed with the fixture answers.  Treat that result as an operational
smoke result only.
