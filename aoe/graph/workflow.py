"""
graph/workflow.py — LangGraph workflow assembly.

Nodes, conditional routing, and the compiled graph.
Entry point is always AOEHandle (middleware/handle.py), never called directly.

Routing logic:
  analyser → if open_questions is empty AND analyser_approved → code_generator
           → if open_questions is empty AND not approved      → END (await user approval)
           → if open_questions not empty                      → END (await user answers)
"""

from langgraph.graph import StateGraph, END

from agents.analyser import analyser_node
from agents.code_generator import code_generator_node
from agents.debug import debug_node
from agents.input_retrieval import input_retrieval_node
from core.config import load_settings
from core.state import GraphState
from solver.runner import solver_node
from tests.dummy.workflow_dummies import dummy_analyser_node, dummy_code_generator_node


_settings = load_settings()

_analyser = dummy_analyser_node if _settings.use_dummy_analyser else analyser_node


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

#regeneration router
def _route_after_solver(state: GraphState) -> str:
    """If error after solver execution, route to debug node"""
    error = state.get("last_execution_error")
    if error:
        return "debug"
    return "explainer"

def _route_after_debug(state: GraphState) -> str:
    """
    Route after debug classification.
    
    Error type → Recovery action mapping:
    - syntax_error, runtime_error, modeling_error → code_generator
    - max_retries_exceeded → explainer (give up, show result)
    - unknown_error → explainer (fallback)
    """
    error_type = state.get("last_error_type")
    
    # CRITICAL: If max retries exceeded, STOP regeneration and end gracefully
    if error_type == "max_retries_exceeded":
        return "explainer"
    
    # Route code errors to code_generator for LLM regeneration
    code_gen_errors = ["syntax_error", "runtime_error", "modeling_error"]
    if error_type in code_gen_errors:
        return "code_generator"
    else:
        return "explainer"  # unknown or fallback


# Graph assembly


_builder = StateGraph(GraphState)

_builder.add_node("analyser", _analyser)
_builder.add_node("input_retrieval", input_retrieval_node)
_builder.add_node("code_generator", _code_generator)
_builder.add_node("solver", solver_node)
_builder.add_node("debug", debug_node)
_builder.add_node("explainer", explainer_node)

_builder.set_entry_point("analyser")
_builder.add_conditional_edges("analyser", _route_after_analyser)
_builder.add_conditional_edges("input_retrieval", _route_after_input_retrieval)
_builder.add_edge("code_generator", "solver")
_builder.add_conditional_edges("solver", _route_after_solver)
_builder.add_conditional_edges("debug", _route_after_debug)

_builder.add_edge("explainer", END)

compiled_graph = _builder.compile()
