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
    #for windows 
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Activate it (run this every time you open a new terminal)
source .venv/bin/activate
    #for windows 
    .venv\Scripts\activate
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
cd aoe
source .venv/bin/activate
python3 -m pytest tests/test_llm_connection.py -v -s

#for windows
python -m pytest tests/test_llm_connection.py -v -s
```

### 5. Run the CLI

```bash
python3 -m ui.cli.main
#for windows
python -m ui.cli.main
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
#for windows
python -m ui.gradio.app

```

### 8. Dummy CSV files for input retrieval testing

Pre-built CSV fixtures live in `tests/dummy/` for the sample transportation problem
(warehouses: ist, ank, izm — stores: kad, üsk, bey):

| File    | Parameter           | Shape     | Contents                            |
|---------|---------------------|-----------|-------------------------------------|
| `c.csv` | Shipping cost       | 3×3 (I×J) | Labeled rows & cols — header row/col auto-stripped by the parser |
| `s.csv` | Supply at warehouse | 1-D (I)   | `150, 175, 200` — one value per row |
| `d.csv` | Demand at store     | 1-D (J)   | `120, 130, 100` — one value per row |

Values are balanced (total supply 525 ≥ total demand 350) so the LP will be feasible.

When the CLI prompts `Path to CSV for c:`, enter:

```
tests/dummy/c.csv
```

Same pattern for `s` and `d`.

### 9. Another sample 

I run a factory that produces 2 products: chairs and tables. Each product requires labor hours and wood. I have limited labor and wood available per week. Each chair earns $25 profit and each table earns $40 profit. I want to maximize total weekly profit. The number of chairs and tables must be whole numbers.
                                                                                                                      
  ---             
  When asked for data:

  ┌────────────────────────────────────────────────┬─────────────────────────────────────────────────────────────┐
  │                     Prompt                     │                           Answer                            │
  ├────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
  │ How many Products (P)?                         │ chair, table                                                │
  ├────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
  │ How many Resources (R)?                        │ labor, wood                                                 │    
  ├────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
  │ Profit per product (profit)                    │ chair = 25, table = 40                                      │    
  ├────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────┤    
  │ Available resource per week (available)        │ labor = 120, wood = 80                                      │
  ├────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────┤    
  │ Resource required per unit (usage) — 2×2 table │ chair needs: labor=2, wood=1 / table needs: labor=4, wood=3 │
  └────────────────────────────────────────────────┴─────────────────────────────────────────────────────────────┘    
                  