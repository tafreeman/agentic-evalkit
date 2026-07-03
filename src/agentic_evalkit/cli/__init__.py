from agentic_evalkit.cli import (  # noqa: F401  (import mounts subcommands onto app)
    datasets,
    doctor,
    reports,
    runs,
)
from agentic_evalkit.cli.app import app

__all__ = ["app"]
