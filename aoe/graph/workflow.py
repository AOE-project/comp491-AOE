"""
graph/workflow.py — LangGraph workflow assembly.

Nodes, conditional routing, and the compiled graph.
Entry point is always AOEHandle (middleware/handle.py), never called directly.

Routing logic:
  analyser → if open_questions is empty AND analyser_approved → latex_generator
           → if open_questions is empty AND not approved      → END (await user approval)
           → if open_questions not empty                      → END (await user answers)
  latex_generator → input_retrieval (always; pass-through if latex_model already set)
"""

import json
import shutil
from pathlib import Path

from langgraph.graph import StateGraph, END

from agents.analyser import analyser_node
from agents.code_generator import code_generator_node
from agents.input_retrieval import input_retrieval_node
from agents.latex_generator import latex_generator_node
from core.config import load_settings
from core.state import GraphState
from solver.runner import solver_node


_DUMMY_SESSION = Path(__file__).parent.parent / "tests" / "dummy_session"

# Dummy analyser node — no LLM calls, no token usage.
# Activate by setting USE_DUMMY_ANALYSER=true in your .env file.

def dummy_analyser_node(state: GraphState) -> dict:
    """
    Replacement for analyser_node for UI development.
    Loads the MILP model from tests/dummy_session/milp_model.json and returns
    a two-turn fake conversation (questions on turn 1, approval-ready on turn 2+)
    so the full Gradio flow can be exercised without spending OpenAI tokens.
    """
    if state.get("analyser_approved"):
        return {}

    turn = state.get("iteration_count", 0)
    base_model = json.loads((_DUMMY_SESSION / "milp_model.json").read_text(encoding="utf-8"))

    if turn == 0:
        return {
            "milp_model": base_model,
            "analyser_output": {"milp_model": base_model},
            "open_questions": [
                "Can each warehouse ship to every store, or are some routes unavailable?",
                "Should all store demand be fully satisfied, or is partial fulfilment acceptable?",
            ],
            "confirmed_assumptions": [],
            "unconfirmed_assumptions": [
                "All warehouse-store routes are available.",
                "All store demand must be fully satisfied.",
            ],
            "analysis_summary": "We are minimising total shipping cost from warehouses to stores, subject to supply and demand constraints.",
            "technical_summary": "min sum_{i,j} c[i][j]*x[i][j]  s.t. supply and demand constraints.",
            "iteration_count": 1,
            "analyser_approved": False,
        }

    return {
        "milp_model": base_model,
        "analyser_output": {"milp_model": base_model},
        "open_questions": [],
        "confirmed_assumptions": [
            "All warehouse-store routes are available.",
            "All store demand must be fully satisfied.",
        ],
        "unconfirmed_assumptions": [],
        "analysis_summary": "We are minimising total shipping cost from warehouses to stores, subject to supply and demand constraints.",
        "technical_summary": "min sum_{i,j} c[i][j]*x[i][j]  s.t. supply and demand constraints.",
        "iteration_count": turn + 1,
        "analyser_approved": False,
    }


_settings = load_settings()

_analyser = dummy_analyser_node if _settings.use_dummy_analyser else analyser_node


def dummy_latex_generator_node(state: GraphState) -> dict:
    """
    Replacement for latex_generator_node for UI development.
    Loads the LaTeX body from tests/dummy_session/latex_body.tex and copies
    the sample PNG into the actual session folder (mirrors what the real node
    does so SessionLogger finds it in the right place).
    Activate with USE_DUMMY_LATEX_GENERATOR=true in your .env file.
    """
    if state.get("latex_model"):
        return {}

    latex_body  = (_DUMMY_SESSION / "latex_body.tex").read_text(encoding="utf-8")
    session_dir = Path(__file__).parent.parent / "sessions" / state.get("session_id", "dummy")
    session_dir.mkdir(parents=True, exist_ok=True)
    png_dest    = session_dir / "milp_formulation.png"
    shutil.copy2(_DUMMY_SESSION / "milp_formulation.png", png_dest)
    return {"latex_model": latex_body, "latex_png_path": str(png_dest)}


_latex_generator = (
    dummy_latex_generator_node
    if _settings.use_dummy_latex_generator
    else latex_generator_node
)


def dummy_code_generator_node(state: GraphState) -> dict:
    """
    Replacement for code_generator_node for UI / integration development.
    Loads actual runnable gurobipy code from tests/dummy_session/generated_code.py
    so the solver can execute it and produce real output.
    Activate with USE_DUMMY_CODE_GENERATOR=true in your .env file.
    """
    code = (_DUMMY_SESSION / "generated_code.py").read_text(encoding="utf-8")
    return {"generated_code": code}


_code_generator = (
    dummy_code_generator_node
    if _settings.use_dummy_code_generator
    else code_generator_node
)


def explainer_node(state: GraphState) -> dict:
    print("[Explainer] Generating explanation...")
    return {"explanation": "# to be implemented"}



# Routing


def _route_after_analyser(state: GraphState) -> str:
    """
    Advance to latex_generator when the user approves, or automatically during
    re-optimisation (is_regeneration=True) when the analyser has no open questions.
    Otherwise return END so AOEHandle can surface questions or the summary.
    """
    if state.get("analyser_approved"):
        return "latex_generator"
    if state.get("is_regeneration") and not state.get("open_questions"):
        return "latex_generator"
    return END


def _route_after_explainer(state: GraphState) -> str:
    """
    Reserved for future routing (e.g. loop back to analyser for multi-step fixes).
    Currently always terminates — AOEHandle picks up the next user message.
    """
    return END


def _route_after_input_retrieval(state: GraphState) -> str:
    """
    Advance to code_generator only when all data has been collected
    (current_input_spec is empty/None).  Otherwise return END to await
    the next user message.
    """
    spec = state.get("current_input_spec")
    if not spec:
        return "code_generator"
    return END



# Graph assembly


_builder = StateGraph(GraphState)

_builder.add_node("analyser", _analyser)
_builder.add_node("latex_generator", _latex_generator)
_builder.add_node("input_retrieval", input_retrieval_node)
_builder.add_node("code_generator", _code_generator)
_builder.add_node("solver", solver_node)
_builder.add_node("explainer", explainer_node)

_builder.set_entry_point("analyser")
_builder.add_conditional_edges("analyser", _route_after_analyser)
_builder.add_edge("latex_generator", "input_retrieval")
_builder.add_conditional_edges("input_retrieval", _route_after_input_retrieval)
_builder.add_edge("code_generator", "solver")
_builder.add_edge("solver", "explainer")
_builder.add_conditional_edges("explainer", _route_after_explainer)

compiled_graph = _builder.compile()
