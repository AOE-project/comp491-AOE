"""
ui/cli/main.py — Typer CLI (CLIApp) and Rich terminal formatters.

CLIApp defines a run command that starts an interactive multi-turn session,
passing user messages to AOEHandle.run() and printing questions and summaries.

Run with:
    python3 -m ui.cli.main
"""

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from middleware.handle import AOEHandle

app = typer.Typer()
console = Console()


# ── Formatting helpers ────────────────────────────────────────────────────────

def _print_prompt(prompt: str) -> None:
    console.print(Panel(prompt, border_style="cyan", padding=(0, 1)))


def _print_error(error: str) -> None:
    console.print(f"  [bold red]✗ {error}[/bold red]")


def _print_collected(label: str, value: str) -> None:
    console.print(f"  [bold green]✔[/bold green] {label}: [dim]{value}[/dim]")


# ── Input retrieval loop ──────────────────────────────────────────────────────

def _run_input_retrieval(handle: AOEHandle, state: dict) -> dict:
    """
    Drive the input_retrieval node to completion.

    set_size  — prompted interactively (count or named labels).
    param_data — user provides a file path to a CSV file.
    """
    console.print("\n[bold cyan]Data Collection[/bold cyan]")
    console.print("[dim]Answer each prompt to provide the data needed for the model.[/dim]\n")

    while True:
        spec      = (state or {}).get("current_input_spec") or {}
        spec_type = spec.get("type", "")

        if not spec_type:
            # All data collected — exit the loop
            break

        prompt = spec.get("prompt", "")
        error  = spec.get("error")

        if error:
            _print_error(error)

        _print_prompt(prompt)

        # ── set_size: free-text answer ────────────────────────────────────
        if spec_type == "set_size":
            set_name = spec.get("set_name", "set")
            answer   = typer.prompt(f"  {set_name}")
            state    = handle.run(answer, state)

        # ── param_data: CSV file path ─────────────────────────────────────
        elif spec_type == "param_data":
            param_name = spec.get("param_name", "parameter")
            shape      = spec.get("shape", [])
            row_labels = spec.get("row_labels", [])
            col_labels = spec.get("col_labels") or []

            # Print a compact shape reminder
            if col_labels:
                dim_hint = f"{len(row_labels)} × {len(col_labels)}  ({' × '.join(shape)})"
            else:
                dim_hint = f"{len(row_labels)} value(s)  ({shape[0] if shape else '?'})"

            console.print(f"  [dim]Expected shape: {dim_hint}[/dim]")
            if row_labels:
                row_preview = ", ".join(row_labels[:5]) + ("…" if len(row_labels) > 5 else "")
                console.print(f"  [dim]Rows : {row_preview}[/dim]")
            if col_labels:
                col_preview = ", ".join(col_labels[:5]) + ("…" if len(col_labels) > 5 else "")
                console.print(f"  [dim]Cols : {col_preview}[/dim]")

            while True:
                raw_path = typer.prompt(f"  Path to CSV for {param_name}")
                path     = Path(raw_path.strip())
                if not path.exists():
                    _print_error(f"File not found: {path}")
                    continue
                if not path.is_file():
                    _print_error(f"Not a file: {path}")
                    continue
                break

            _print_collected(param_name, str(path))
            state = handle.run(str(path), state)

            # Check immediately if the node reported a validation error
            new_spec  = (state or {}).get("current_input_spec") or {}
            new_error = new_spec.get("error")
            if new_error:
                # Loop will print the error on the next iteration
                continue

        else:
            # Unknown spec type — skip by sending an empty string
            state = handle.run("", state)

    console.print("\n[bold green]All data collected.[/bold green]\n")
    return state


# ── Main command ──────────────────────────────────────────────────────────────

@app.command()
def run(
    resume: str = typer.Option(None, "--resume", "-r", help="Session ID to resume"),
):
    console.print("\n[bold cyan]AOE — Automated Optimization Engineer[/bold cyan]\n")

    if resume:
        try:
            handle, state = AOEHandle.resume(resume)
            console.print(f"[bold green]Resumed session:[/bold green] {resume}\n")
        except FileNotFoundError as exc:
            console.print(f"[bold red]Error:[/bold red] {exc}")
            raise typer.Exit(1)

        # If input retrieval is already done, skip straight to code generation
        spec = (state or {}).get("current_input_spec") or {}
        if not spec and state.get("analyser_approved"):
            console.print("[bold]Input data already collected — continuing to code generation.[/bold]")
            state = handle.run("__resume__", state)
            console.print("[bold]Done.[/bold]")
            return

        # If analyser not yet approved, re-enter the Q&A loop below
        if not state.get("analyser_approved"):
            console.print("[dim]Resuming analyser dialogue…[/dim]\n")
    else:
        handle = AOEHandle()
        state  = None
        problem = typer.prompt("Describe your optimisation problem")
        state   = handle.run(problem, state)

    # ── Analyser Q&A loop (skipped on resume if already approved) ────────
    if not state.get("analyser_approved"):
        while True:
            questions = state.get("open_questions", [])
            summary   = state.get("analysis_summary", "")

            if questions:
                console.print("\n[bold yellow]Questions:[/bold yellow]")
                for q in questions:
                    console.print(f"  • {q}")
                answer = typer.prompt("\nYour answer")
                state  = handle.run(answer, state)

            else:
                console.print(f"\n[bold green]Model summary:[/bold green]\n{summary}")
                approved = typer.confirm("\nDoes this look correct?")
                if not approved:
                    feedback = typer.prompt("What would you like to change?")
                    state    = handle.run(feedback, state)
                    continue

                # User approved — advance to input retrieval
                state["analyser_approved"] = True
                state = handle.run("__approved__", state)
                break

    # ── Input retrieval loop (skipped on resume if already done) ─────────
    if (state or {}).get("current_input_spec"):
        state = _run_input_retrieval(handle, state)

    # ── Code generation result ────────────────────────────────────────────
    generated_code = (state.get("generated_code") or "").strip()
    if generated_code:
        console.print("[bold cyan]Generated Gurobi Script:[/bold cyan]")
        console.print(Panel(generated_code, border_style="dim", padding=(0, 1)))

        syntax_error = state.get("code_syntax_error")
        if syntax_error:
            console.print(f"\n[bold red]Syntax error:[/bold red] {syntax_error}")
    else:
        console.print("[bold red]Code generation failed — no script produced.[/bold red]")
        raise typer.Exit(1)

    # ── Solver result ─────────────────────────────────────────────────────
    # Solver has already run in the same graph invocation that completed
    # input retrieval (when the last required CSV was provided).
    solver_result = state.get("solver_result") or {}
    if solver_result:
        status    = (solver_result.get("status") or "").lower()
        obj_value = solver_result.get("objective_value")
        variables = solver_result.get("variables") or {}
        iis_path  = solver_result.get("iis_path")
        stderr    = (solver_result.get("stderr") or "").strip()

        if status == "optimal":
            console.print("\n[bold green]Solver Result: Optimal[/bold green]")
            lines = []
            if obj_value is not None:
                lines.append(f"Objective: {obj_value}")
            if variables:
                lines.append("")
                lines.append("Variables:")
                for name, value in variables.items():
                    lines.append(f"  {name} = {value}")
            else:
                lines.append("(no non-zero decision variables)")
            console.print(Panel("\n".join(lines), border_style="green", padding=(0, 1)))
        elif status == "infeasible":
            console.print("\n[bold red]Solver Result: Infeasible[/bold red]")
            note = "Model has no feasible solution."
            if iis_path:
                note += f"\nIIS written to: {iis_path}"
            console.print(Panel(note, border_style="red", padding=(0, 1)))
        elif status == "unbounded":
            console.print("\n[bold yellow]Solver Result: Unbounded[/bold yellow]")
            console.print(Panel(
                "Objective is unbounded — no finite optimum exists.",
                border_style="yellow", padding=(0, 1),
            ))
        else:
            console.print(f"\n[bold red]Solver Error ({status or 'unknown'}):[/bold red]")
            console.print(Panel(stderr or "No diagnostic output.", border_style="red", padding=(0, 1)))

if __name__ == "__main__":
    app()
