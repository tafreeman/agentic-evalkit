# agentic-evalkit

`agentic-evalkit` is a standalone evaluation toolkit for agentic systems. It combines dynamic dataset discovery, typed evaluation contracts, benchmark-valid grading, calibrated judges, statistical reporting, and a developer-friendly Python API and CLI.

The repository is currently in design. See [the architecture specification](docs/superpowers/specs/2026-07-02-agentic-evalkit-design.md).

## Identity

- Distribution and repository: `agentic-evalkit`
- Python package: `agentic_evalkit`
- CLI: `agentic-evalkit`

## Repository boundary

This project does not modify or import Agentic Runtime Platform or ExecutionKit internals. Those systems may be evaluated through stable callable, subprocess, or HTTP target adapters.
