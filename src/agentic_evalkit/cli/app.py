import typer

from agentic_evalkit import __version__

app = typer.Typer(no_args_is_help=True, help="Evaluate agentic systems with reproducible evidence.")


@app.callback(invoke_without_command=True)
def main(
    version: bool = typer.Option(False, "--version", help="Show the installed version."),
) -> None:
    if version:
        typer.echo(__version__)
        raise typer.Exit()
