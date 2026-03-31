# aoe — Automated Optimization Engineer

A multi-agent LLM system that automates Mixed-Integer Linear Programming (MILP)
optimization end-to-end. Four specialized agents (Analyser, CodeGenerator, Debug,
Explainer) run in a LangGraph workflow powered by gpt-5.4-nano, producing and executing
Gurobi solver scripts from natural-language problem descriptions.

## Project Structure

```
aoe/
├── prompts/          # Prompt templates per agent (.txt)
├── core/
│   ├── config.py     # Settings (Pydantic) + get_llm_client()
│   └── state.py      # GraphState, MILPModel, SolverResult (Pydantic) + exceptions
├── agents/
│   ├── base.py       # BaseAgent (abstract) + PromptRegistry
│   ├── analyser.py
│   ├── code_generator.py
│   ├── debug.py
│   └── explainer.py
├── graph/
│   └── workflow.py   # LangGraph nodes, edges, compiled_graph
├── solver/
│   └── runner.py     # SolverRunner + SolverParser
├── middleware/
│   ├── handle.py     # AOEHandle — sole entry point
│   ├── events.py     # Event dataclasses (AgentStarted, SolverExecuted, ...)
│   └── session.py    # SessionManager
├── tests/
│   └── test_llm_connection.py  # Integration test for LLM API
└── ui/
    ├── cli/
    │   └── main.py   # CLIApp (Typer) + terminal formatters
    └── gradio/
        └── app.py    # GradioApp (Gradio Blocks) + component builders
```

## Getting Started

This project uses uv for dependency management. Make sure it is installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 1. Create and activate the virtual environment

```bash
# Create venv (only needed once)
uv venv .venv

# Activate it (run this every time you open a new terminal)
source .venv/bin/activate
```

### 2. Install dependencies

```bash
# Install the project and all dependencies declared in pyproject.toml
uv pip install -e ".[dev]"
```

If you add a new package to `pyproject.toml`, re-run the same command to sync.

### 3. Configure environment

```bash
cp .env.example .env
# Open .env and fill in OPENAI_API_KEY and GUROBI_LICENSE_PATH
```

### 4. Run the tests

```bash
# From the aoe/ directory with the venv activated:
python3 -m pytest tests/test_llm_connection.py -v -s
```

### 5. Run the CLI

```bash
python3 -m ui.cli.main
```

can try with question:
I want to minimise the total shipping cost from three warehouses to four retail stores. Each warehouse has a limited supply and each store has a  fixed demand. I need to decide how much to ship from each warehouse to each store.

### 6. UI development without API calls (dummy mode)

Set `USE_DUMMY_ANALYSER=true` in `.env` to replace the Analyser agent with a hardcoded stub.
The stub simulates two dialogue turns — returning questions on turn 1 and a completed model on
turn 2 — so the full CLI/Gradio conversation flow can be exercised without spending OpenAI tokens.
Useful when building or testing the UI layer independently of the LLM.

### 7. Launch the Gradio UI

```bash
python3 -m ui.gradio.app
```