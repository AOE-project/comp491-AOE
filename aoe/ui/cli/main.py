"""
ui/cli/main.py — Typer CLI (CLIApp) and Rich terminal formatters.

CLIApp defines a run command that starts an interactive multi-turn session,
passing user messages to AOEHandle.run() and printing questions and summaries.

Run with:
    python3 -m ui.cli.main
"""

import typer
from rich.console import Console
from middleware.handle import AOEHandle

app = typer.Typer()
console = Console()


@app.command()
def run():
    console.print("\n[bold cyan]AOE — Automated Optimization Engineer[/bold cyan]\n")
    handle = AOEHandle()
    state = None

    problem = typer.prompt("Describe your optimisation problem")
    state = handle.run(problem, state)

    while True:
        questions = state.get("open_questions", [])
        summary = state.get("analysis_summary", "")

        if questions:
            console.print("\n[bold yellow]Questions:[/bold yellow]")
            for q in questions:
                console.print(f"  • {q}")
            answer = typer.prompt("\nYour answer")
            state = handle.run(answer, state)

        else:
            console.print(f"\n[bold green]Model summary:[/bold green]\n{summary}")
            approved = typer.confirm("\nDoes this look correct?")
            if approved:
                state["analyser_approved"] = True
                console.print("\n[bold]Model approved. Next steps (code generation) coming soon.[/bold]")
                console.print(f"\n[bold green]Analysis:[/bold green]\n{summary}")
                break
            else:
                feedback = typer.prompt("What would you like to change?")
                state = handle.run(feedback, state)


if __name__ == "__main__":
    app()
