"""
graph/workflow.py — LangGraph workflow assembly.

Nodes, conditional routing, and the compiled graph.
Entry point is always AOEHandle (middleware/handle.py), never called directly.

Routing logic:
  analyser        → if analyser_approved → input_retrieval
                  → otherwise            → END (await answers or approval)
  input_retrieval → if current_input_spec empty → code_generator
                  → otherwise                   → END (await next data item)
"""

from langgraph.graph import StateGraph, END

from agents.analyser import analyser_node
from agents.input_retrieval import input_retrieval_node
from core.config import load_settings
from core.state import GraphState


# Dummy analyser node — no LLM calls, no token usage.
# Activate by setting USE_DUMMY_ANALYSER=true in your .env file.

def dummy_analyser_node(state: GraphState) -> dict:
    """
    Replacement for analyser_node for UI development.
    Returns a hardcoded minimal MILP model after two fake turns so the
    Gradio/CLI developer can exercise the full conversation flow without
    spending OpenAI tokens.

    Turn 1: returns open questions.
    Turn 2+: clears questions and marks the model ready for approval.
    """
    # Model already approved — pass through so the router advances.
    if state.get("analyser_approved"):
        return {}

    turn = state.get("iteration_count", 0)

    base_model = {
        "sets": [{"name": "I", "description": "Warehouses"}, {"name": "J", "description": "Stores"}],
        "tuples": [{"name": "IJ", "description": "Warehouse-store pairs", "component_sets": ["I", "J"]}],
        "parameters": [
            {"name": "c", "description": "Shipping cost", "index_sets": ["I", "J"], "shape": ["I", "J"], "data_key": "cost"},
            {"name": "s", "description": "Supply at warehouse", "index_sets": ["I"], "shape": ["I"], "data_key": "supply"},
            {"name": "d", "description": "Demand at store", "index_sets": ["J"], "shape": ["J"], "data_key": "demand"},
        ],
        "variables": [{"name": "x", "description": "Units shipped", "type": "continuous", "index_sets": ["I", "J"], "lb": 0, "ub": None}],
        "objective": {"sense": "minimize", "expression": "sum_{i in I} sum_{j in J} c[i][j] * x[i][j]"},
        "constraints": [
            {"name": "supply", "description": "Supply limit", "expression": "sum_{j in J} x[i][j] <= s[i]  for all i in I", "index_sets": ["I"]},
            {"name": "demand", "description": "Demand fulfilment", "expression": "sum_{i in I} x[i][j] >= d[j]  for all j in J", "index_sets": ["J"]},
        ],
    }

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


_analyser = dummy_analyser_node if load_settings().use_dummy_analyser else analyser_node


# Placeholder nodes 


def code_generator_node(state: GraphState) -> dict:
    print("[CodeGenerator] Generating Gurobi script from MILP model...")
    return {"generated_code": "# to be implemented"}


def explainer_node(state: GraphState) -> dict:
    print("[Explainer] Generating explanation...")
    return {"explanation": "# to be implemented"}



# Routing


def _route_after_analyser(state: GraphState) -> str:
    """
    Advance to input_retrieval only when the user has explicitly approved the model.
    Otherwise return END so AOEHandle can surface questions or the summary to the user
    and wait for the next message.
    """
    if state.get("analyser_approved"):
        return "input_retrieval"
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
_builder.add_node("input_retrieval", input_retrieval_node)
_builder.add_node("code_generator", code_generator_node)
_builder.add_node("explainer", explainer_node)

_builder.set_entry_point("analyser")
_builder.add_conditional_edges("analyser", _route_after_analyser)
_builder.add_conditional_edges("input_retrieval", _route_after_input_retrieval)
_builder.add_edge("code_generator", "explainer")
_builder.add_edge("explainer", END)

compiled_graph = _builder.compile()
