"""
tests/test_analyser.py — Integration test for the Analyser agent node.
Sends a simple MILP problem description and asserts the agent returns a
structurally valid response. Hits the real API; requires .env to be set.

Run with:
    python3 -m pytest tests/test_analyser.py -v -s
"""

from agents.analyser import analyser_node

_PROBLEM = (
    "I want to minimise the total shipping cost from three warehouses to four "
    "retail stores. Each warehouse has a limited supply and each store has a "
    "fixed demand. I need to decide how much to ship from each warehouse to "
    "each store."
)


def _base_state() -> dict:
    return {
        "session_id": "test-session",
        "problem_description": _PROBLEM,
        "history": [{"role": "user", "content": _PROBLEM}],
        "iteration_count": 0,
        "analyser_output": None,
        "milp_model": {},
        "confirmed_assumptions": [],
        "unconfirmed_assumptions": [],
        "open_questions": [],
        "analysis_summary": "",
        "technical_summary": "",
        "analyser_approved": False,
        "raw_data": {},
        "generated_code": "",
        "solver_result": {},
        "explanation": "",
        "token_usage": {},
    }


def test_analyser_first_turn():
    state = _base_state()
    result = analyser_node(state)

    print("\n--- open_questions ---")
    for q in result.get("open_questions", []):
        print(" •", q)

    print("\n--- analysis_summary ---")
    print(result.get("analysis_summary", ""))

    print("\n--- milp_model keys ---")
    print(list(result.get("milp_model", {}).keys()))

    # Basic structural assertions
    assert result.get("iteration_count") == 1
    assert isinstance(result.get("milp_model"), dict)
    assert isinstance(result.get("open_questions"), list)
    assert isinstance(result.get("analysis_summary"), str)
    assert isinstance(result.get("technical_summary"), str)
    assert result.get("analysis_summary") != ""


def test_analyser_second_turn():
    """Simulate a second turn: user answers the first batch of questions."""
    state = _base_state()

    # First turn
    first_result = analyser_node(state)
    state.update(first_result)

    # User answers the open questions generically
    user_answer = (
        "Yes, each warehouse can ship to any store. "
        "The objective is to minimise total cost. "
        "All shipment quantities must be non-negative."
    )
    state["history"].append({"role": "user", "content": user_answer})

    # Second turn
    second_result = analyser_node(state)

    print("\n--- second turn open_questions ---")
    for q in second_result.get("open_questions", []):
        print(" •", q)

    print("\n--- confirmed_assumptions ---")
    for a in second_result.get("confirmed_assumptions", []):
        print(" •", a)

    assert second_result.get("iteration_count") == 2
    assert isinstance(second_result.get("confirmed_assumptions"), list)
