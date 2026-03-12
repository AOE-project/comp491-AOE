"""
graph/workflow.py — Workflow.

Wraps agents as LangGraph nodes, defines conditional routing, and compiles
the graph into compiled_graph. Invoked only through AOEHandle.
"""

from langgraph.graph import StateGraph, END
from core.state import GraphState


# Dummy nodes

def analyser_node(state: GraphState) -> dict:
    print("[Analyser] Extracting MILP structure from problem description...")
    return {"milp_model": {"sets": [], "variables": [], "constraints": [], "objective": None}}


def code_generator_node(state: GraphState) -> dict:
    print("[CodeGenerator] Generating Gurobi script from MILP model...")
    return {"generated_code": "# dummy gurobi script\nprint('No real code yet.')"}


def explainer_node(state: GraphState) -> dict:
    print("[Explainer] Generating explanation...")
    return {"explanation": "Dummy explanation: the model was analysed and code was generated."}


# Graph assembly

_builder = StateGraph(GraphState)

_builder.add_node("analyser", analyser_node)
_builder.add_node("code_generator", code_generator_node)
_builder.add_node("explainer", explainer_node)

_builder.set_entry_point("analyser")
_builder.add_edge("analyser", "code_generator")
_builder.add_edge("code_generator", "explainer")
_builder.add_edge("explainer", END)

compiled_graph = _builder.compile()