"""Dummy workflow nodes used for development and integration testing."""

from core.state import GraphState


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
    # Always return valid code - solver_node handles error injection based on
    # TEST_ERROR_INJECTION setting and regeneration_attempts counter

    raw_data = state.get("raw_data", {})

    # Only increment regeneration_attempts if this is a regeneration (not initial generation)
    last_error_type = state.get("last_error_type")
    code_errors = ["syntax_error", "runtime_error", "modeling_error"]
    is_regenerating = last_error_type in code_errors

    if is_regenerating:
        print(
            f"[DEBUG] dummy_code_generator_node: is_regenerating=True, "
            f"error_type={last_error_type}, "
            f"regeneration_attempts={state.get('regeneration_attempts', 0)}"
        )
        print(f"[REGENERATION] Attempting fix for {last_error_type.upper()} error")

    return {
        "generated_code": (
            "# [DUMMY] Code generation skipped - USE_DUMMY_CODE_GENERATOR=true\n"
            "import gurobipy as gp\n"
            "from gurobipy import GRB\n\n"
            f"# raw_data keys available: {list(raw_data.keys())}\n\n"
            "m = gp.Model('dummy')\n"
            "m.Params.OutputFlag = 0\n"
            "m.optimize()\n"
            "result = {'status': 'dummy', 'objective_value': None, 'variables': {}}\n"
            "print(result)\n"
        ),
        "regeneration_attempts": state.get("regeneration_attempts", 0) + 1 if is_regenerating else state.get("regeneration_attempts", 0),
    }
