# Automated Optimization Engineer (AOE)
> **A multi-agent LLM system that turns natural-language optimization problems into structured MILP models, executable Gurobi code, and solver-validated solutions.**
Developed as a **COMP 491 Computer Engineering Design Project**.
AOE automates the workflow:
**problem description → clarification → MILP structure → Gurobi code → solver execution → automatic debugging → explanation**
The goal is not just to generate optimization code with an LLM, but to build a **structured, reliable, solver-grounded optimization engineering pipeline**.
---
## Project at a Glance
- **4 specialized LLM agents**
- **LangGraph-based orchestration**
- **Structured JSON communication** between stages
- **Gurobi-backed validation**
- **Automatic error recovery and regeneration**
- **Strict separation of reasoning and numerical instance data**
- **Gradio web interface + CLI**
- **100% end-to-end completion** on the reported 20-problem benchmark
- **80% execution rate**
- **90% optimization success rate**
- Target workflow time: **under 15 minutes**
- Measured runtime cost: **below $0.01 per modeling cycle**
---
## The Problem
Mixed-Integer Linear Programming (**MILP**) is widely used in supply chain, production planning, scheduling, logistics, resource allocation, and energy systems.
The solver is usually not the hardest part.
The real challenge is translating an ambiguous real-world problem into a correct model with:
- sets and indices
- parameters
- decision variables
- objective functions
- constraints
- valid dimensions
- executable solver code
This process normally requires optimization expertise, repeated clarification, formulation, debugging, and validation.
AOE is designed to automate that workflow while keeping the **solver—not the LLM—as the final computational authority**.
---
## System Architecture
```text
Natural-Language Problem
          ↓
    Analyser Agent
 clarification + MILP structure
          ↓
 Structured JSON + Validation
          ↓
     Code Generator
      Python/Gurobi
          ↓
      Gurobi Solver
       ↙       ↘
    Error       Success
      ↓            ↓
Debug/Regenerate  Chat Agent
      ↓            ↓
 regenerate     Explain / Refine
```
The workflow is coordinated through a shared **LangGraph state**.
---
## Agent 1 — Analyser
The Analyser converts a natural-language problem into a formal optimization structure.
It extracts:
- index sets and tuple sets
- parameters and dimensions
- decision variables and domains
- objective direction and structure
- linear constraints
Unlike one-shot prompting, AOE uses **iterative clarification**.
If a requirement is ambiguous, the system asks targeted questions before code generation.
Example:
> Can one worker be assigned to multiple shifts on the same day, or at most one?
This prevents unclear assumptions from silently propagating into the model.
---
## Structured Intermediate Representation
The Analyser does not pass free-form text directly to the next stage.
It creates a structured JSON representation such as:
```json
{
  "sets": {"I": ["..."], "J": ["..."]},
  "parameters": {"c": {"domain": "IxJ", "type": "float"}},
  "variables": {"x_ij": {"domain": "IxJ", "type": "continuous"}},
  "objective": "...",
  "constraints": ["..."]
}
```
This JSON acts as a **contract between agents** and reduces errors such as undefined sets, missing parameters, index mismatches, and inconsistent dimensions.
---
## Agent 2 — Code Generator
The Code Generator transforms the validated model structure into executable **Python/Gurobi** code.
It handles:
- model creation
- variable definition
- objective declaration
- constraint generation
- optimization call
- structured result export
Before execution, generated code is checked for syntax and structural consistency.
---
## Numerical Data Isolation
AOE deliberately separates LLM reasoning from finalized instance-level numerical data.
Values such as costs, demands, capacities, supplies, and probabilities are loaded from trusted external sources such as CSV files.
The LLM mainly reasons about:
- names
- domains
- dimensions
- set relationships
- mathematical structure
This reduces the risk of hallucinated or altered numerical values.
---
## Five-Stage Validation Pipeline
AOE uses multiple validation layers:
1. **Schema Validation** — checks required fields and types
2. **Dimension Consistency** — checks index alignment
3. **Code Compilation** — checks generated Python syntax
4. **Solver Execution** — uses Gurobi as execution-grounded validation
5. **Debugging & Regeneration** — feeds failures back into the pipeline
This creates a closed-loop system rather than a one-shot code generator.
---
## Agent 3 — Debug / Regeneration
When execution fails, the system:
1. captures the error
2. classifies the failure
3. stores attempt history
4. creates structured repair context
5. routes the problem back to code generation
6. executes the corrected model again
Handled failure classes include syntax, runtime, modeling, and unknown errors.
A first-pass failure can therefore still become a successful end-to-end run.
---
## Agent 4 — Chat Agent
After solver execution, the Chat Agent converts structured optimization output into a human-readable explanation.
It can interpret:
- solver status
- objective value
- non-zero variable assignments
- execution information
- infeasibility information when available
It also supports post-solution refinement.
For example, a user can request a constraint change and trigger a new modeling cycle.
---
## Technology Stack
| Technology | Role |
|---|---|
| **Python** | Core implementation |
| **LangGraph** | Multi-agent orchestration |
| **Gurobi** | MILP solver and validation backend |
| **JSON Schema** | Structured inter-agent protocol |
| **Gradio** | Local web interface |
| **GPT-5o-nano** | LLM backend used in the reported system |
---
## Evaluation
AOE is evaluated with three end-to-end metrics:
- **Completion Rate** — whether the full pipeline reaches a terminal result
- **Execution Rate** — whether generated models produce an executable solver outcome
- **Optimization Success Rate** — whether the final solver status is `OPTIMAL`
The metrics evaluate the complete autonomous system, including recovery—not only first-pass generation.
---
## Reported Benchmark Results
The project evaluated the first **20 problems** from an NL4Opt-style benchmark derived from the **NeurIPS 2022 NL4Opt dataset**.
| Metric | Result |
|---|---:|
| Analyser Executable Outputs | **100%** |
| End-to-End Completion Rate | **100%** |
| Execution Rate | **80%** |
| Optimization Success Rate | **90%** |
The main remaining weakness is execution-level reliability on a subset of cases, while the recovery loop improves final end-to-end outcomes.
---
## AOE vs. Manual Workflow
| Task | Manual Expert | AOE |
|---|---:|---:|
| Problem clarification | 30–60 min | 5–10 min |
| Model formulation | 60–120 min | ~2 min |
| Code generation & debugging | 120–240 min | <1 min + regeneration |
| **Total modeling time** | **3.5–7 hours** | **<15 minutes** |
| Validation | Manual | Solver-grounded |
| Error recovery | Manual | Automatic regeneration |
| Cost | Expert labor | **< $0.01/cycle** in reported setup |
The aim is to reduce repetitive modeling effort and lower the barrier to optimization—not to remove the need for expert oversight.
---
## User Experience
A typical workflow is:
1. describe the optimization problem
2. answer clarification questions
3. review/refine constraints
4. provide numerical data
5. generate and solve the model
6. inspect the result
7. refine the model conversationally
The project provides both a **Gradio UI** and a **CLI interface**.
---
## Repository
```text
comp491-AOE/
├── aoe/
└── .gitignore
```
GitHub:
`https://github.com/AOE-project/comp491-AOE`
---
## Current Limitations
- focused on linear / mixed-integer linear optimization
- some benchmark cases still fail at execution level
- no full nonlinear-programming support
- limited advanced sensitivity/scenario visualization
- no direct production ERP/database connectors yet
- high-stakes use still requires expert validation
---
## Roadmap
- Non-Linear Programming support
- quadratic objectives
- non-linear constraints
- richer sensitivity analysis
- scenario visualization
- live database connectors
- ERP integration
- broader benchmark evaluation
- stronger execution reliability
- production-grade observability and testing
---
## Why This Project Is Relevant
AOE combines:
- **Large Language Models**
- **Multi-Agent Systems**
- **Operations Research**
- **Mathematical Optimization**
- **Code Generation**
- **Decision-Support Systems**
- **Agent Orchestration**
- **AI Reliability**
> **LLMs structure and automate the modeling workflow; schemas and optimization solvers enforce correctness.**
This makes AOE relevant to AI engineering, agentic systems, operations research, optimization software, decision intelligence, and enterprise automation roles.
---
## Team
**COMP 491 — Computer Engineering Design Project**
- Ayça Biçer
- Begüm Yılmaz
- Ulaş Ardıl Baran
**Advisor:** Mehmet Gönen
**Spring 2026**
