# Each of these modules registers its own commands (e.g. "doctor", "report")
# onto the shared "app" object the moment it is imported, via @app.command()
# decorators inside that module. Nothing here refers to the module names
# directly, so this import exists purely for that side effect -- which is
# also why the linter would otherwise flag it as an unused import (F401).
from agentic_evalkit.cli import (  # noqa: F401
    datasets,
    doctor,
    reports,
    runs,
)
from agentic_evalkit.cli.app import app

__all__ = ["app"]
