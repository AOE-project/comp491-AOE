"""
ui/cli/main.py — Typer CLI (CLIApp) and Rich terminal formatters.

CLIApp defines a run command that starts an interactive session, passing user
messages to AOEHandle.run() and streaming events to the terminal. Supports
--session-id for session resume.
"""

import typer
from rich.console import Console
from middleware.handle import AOEHandle

app = typer.Typer()
console = Console()


@app.command()
def run(problem: str = typer.Argument(..., help="MILP problem description")):
    console.print("\n[bold cyan]AOE — Automated Optimization Engineer[/bold cyan]\n")
    handle = AOEHandle()
    result = handle.run(problem)
    console.print(f"[green]Session:[/green] {result['session_id']}")
    console.print(f"[green]MILP Model:[/green] {result['milp_model']}")
    console.print(f"[green]Generated Code:[/green]\n{result['generated_code']}")
    console.print(f"[green]Explanation:[/green] {result['explanation']}")


if __name__ == "__main__":
    app()
