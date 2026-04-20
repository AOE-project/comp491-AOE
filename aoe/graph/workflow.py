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


_settings = load_settings()

_analyser = dummy_analyser_node if _settings.use_dummy_analyser else analyser_node


def dummy_code_generator_node(state: GraphState) -> dict:
    """
    Replacement for code_generator_node for UI / integration development.
    Returns a minimal but syntactically valid gurobipy stub so the rest of
    the pipeline can be exercised without spending tokens.
    Activate with USE_DUMMY_CODE_GENERATOR=true in your .env file.
    
    Error testing controlled via .env:
      TEST_ERROR_INJECTION: Enable/disable error injection
      TEST_ERROR_TYPE: syntax_error | runtime_error | modeling_error | unknown_error
    
    NOTE: For syntax_error, runtime_error, and modeling_error testing:
    Error injection actually occurs in solver_node, based on regeneration_attempts.
    This node always returns valid (syntactically correct) code.
    """
    # Always return valid code — solver_node handles error injection based on
    # TEST_ERROR_INJECTION setting and regeneration_attempts counter
    
    raw_data = state.get("raw_data", {})
    
    return {
        "generated_code": (
            "# [DUMMY] Code generation skipped — USE_DUMMY_CODE_GENERATOR=true\n"
            "import gurobipy as gp\n"
            "from gurobipy import GRB\n\n"
            f"# raw_data keys available: {list(raw_data.keys())}\n\n"
            "m = gp.Model('dummy')\n"
            "m.Params.OutputFlag = 0\n"
            "m.optimize()\n"
            "result = {'status': 'dummy', 'objective_value': None, 'variables': {}}\n"
            "print(result)\n"
        )
    }


_code_generator = (
    dummy_code_generator_node
    if _settings.use_dummy_code_generator
    else code_generator_node
)


def dummy_regeneration_node(state: GraphState) -> dict:
    """
    Replacement for actual regeneration during debug flow (testing only).
    Mocks regeneration based on error classification.
    
    For code errors: Demonstrates multi-attempt recovery
    - Attempt 1: Still contains error (to test retry loop)
    - Attempt 2+: Error is fixed (recovery succeeds)
    
    For other errors: Immediately fixes
    
    Activate with USE_DUMMY_REGENERATION=true in your .env file.
    """
    error_type = state.get("last_error_type")
    attempts = state.get("debug_attempts", [])
    attempt_count = len(attempts)
    
    # Route based on error type
    code_gen_errors = ["syntax_error", "runtime_error", "modeling_error"]
    
    if error_type in code_gen_errors:
        # Mock: demonstrate multi-attempt recovery
        # Attempt 1: Still broken
        # Attempt 2+: Fixed
        
        regeneration_attempts = (state.get("regeneration_attempts") or 0) + 1
        print(f"[DEBUG] dummy_regeneration: regeneration_attempts={regeneration_attempts}, will_inject_error={(regeneration_attempts < 2)}")
        
        if regeneration_attempts < 2:
            # 1st regeneration: Error continues (to test recovery loop)
            corrected_code = (
                "# [DUMMY REGEN] Attempt 1: Fixing code (still broken)\n"
                "import gurobipy as gp\n"
                "from gurobipy import GRB\n\n"
                "W = ['W1', 'W2', 'W3']\n"
                "S = ['S1', 'S2', 'S3']\n"
                "m = gp.Model('attempt1')\n"
                "m.Params.OutputFlag = 0\n"
                "x = m.addVariable(name='x')  # ← BROKEN: addVariable() doesn't exist\n"
                "m.optimize()\n"
                "result = {'status': 'error'}\n"
                "print(result)\n"
            )
        else:
            # 2nd or later regeneration: NOW FIX
            corrected_code = (
                "# [DUMMY REGEN] Fixed code (attempt successful)\n"
                "import gurobipy as gp\n"
                "from gurobipy import GRB\n\n"
                "W = ['W1', 'W2', 'W3']\n"
                "S = ['S1', 'S2', 'S3']\n"
                "m = gp.Model('fixed_model')\n"
                "m.Params.OutputFlag = 0\n"
                "x = m.addVar(lb=0, ub=10, name='x')  # ← FIXED: addVar() is correct\n"
                "m.setObjective(x, GRB.MAXIMIZE)\n"
                "m.optimize()\n"
                "result = {'status': 'optimal', 'objective_value': float(x.X)}\n"
                "print(result)\n"
            )
        
        return {
            "generated_code": corrected_code,
            "last_execution_error": None,
            "last_error_type": None,
            "regeneration_attempts": regeneration_attempts,
        }
    
    else:
        # unknown_error or other unhandled types - signal failure
        return {
            "last_execution_error": None,
            "last_error_type": None,
        }


_regeneration = (
    dummy_regeneration_node
    if getattr(_settings, "use_dummy_regeneration", False)
    else None  # real regen handled by routing back to agents
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
    - syntax_error, runtime_error, modeling_error → code_generator (LLM regenerates)
    - max_retries_exceeded → explainer (give up, show result)
    - unknown_error → explainer (fallback)
    """
    error_type = state.get("last_error_type")
    
    # CRITICAL: If max retries exceeded, STOP regeneration and end gracefully
    if error_type == "max_retries_exceeded":
        return "explainer"
    
    # If dummy regeneration is enabled, route there instead of individual agents
    if _regeneration:
        return "dummy_regeneration"
    
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

# Add dummy_regeneration node if enabled
if _regeneration:
    _builder.add_node("dummy_regeneration", _regeneration)

_builder.set_entry_point("analyser")
_builder.add_conditional_edges("analyser", _route_after_analyser)
_builder.add_conditional_edges("input_retrieval", _route_after_input_retrieval)
_builder.add_edge("code_generator", "solver")
_builder.add_conditional_edges("solver", _route_after_solver)
_builder.add_conditional_edges("debug", _route_after_debug)

# Route dummy_regeneration back to solver for retry
if _regeneration:
    _builder.add_edge("dummy_regeneration", "solver")

_builder.add_edge("explainer", END)

compiled_graph = _builder.compile()
