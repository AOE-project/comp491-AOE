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
from core.state import GraphState



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
    Advance to code_generator only when the user has explicitly approved the model.
    Otherwise return END so AOEHandle can surface questions or the summary to the user
    and wait for the next message.
    """
    if state.get("analyser_approved"):
        return "code_generator"
    return END



# Graph assembly


_builder = StateGraph(GraphState)

_builder.add_node("analyser", analyser_node)
_builder.add_node("code_generator", code_generator_node)
_builder.add_node("explainer", explainer_node)

_builder.set_entry_point("analyser")
_builder.add_conditional_edges("analyser", _route_after_analyser)
_builder.add_edge("code_generator", "explainer")
_builder.add_edge("explainer", END)

compiled_graph = _builder.compile()
