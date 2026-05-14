"""
agents/code_generator.py — CodeGeneratorAgent.

Translates a validated structural MILP model into a complete, runnable
gurobipy Python script.

Inputs  (from GraphState): milp_model, raw_data
Output  (to  GraphState):  generated_code

The LLM only receives milp_model (structure only).
raw_data is injected as Python literals by _build_data_section() and
prepended to the LLM-generated model code before the final script is stored.
"""

import ast
import json
import re
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from core.config import get_llm_client
from core.state import GraphState
from core.cost_calculator import CostCalculator

_PROMPTS_DIR   = Path(__file__).parent.parent / "prompts"
_SYSTEM_PROMPT = (_PROMPTS_DIR / "code_generator_prompt.txt").read_text(encoding="utf-8")

_MAX_RETRIES = 3

# Fixed Python block appended after the LLM model code. It reads m.status,
# m.objVal, and m.getVars() to write result.json (read by the UI) and
# iis.ilp (on infeasibility) into CWD — which the solver runner sets to
# sessions/<session_id>/.  Keeping this out of the prompt lets the LLM
# focus on modelling instead of result plumbing.
_RESULT_EPILOGUE = '''
# --- Result Section (auto-generated) ---
import json as _aoe_json
import pathlib as _aoe_pathlib

if m.status == GRB.OPTIMAL:
    _aoe_result = {
        "status": "optimal",
        "objective_value": float(m.objVal),
        "variables": {
            v.VarName: float(v.X)
            for v in m.getVars()
            if abs(v.X) > 1e-6
        },
    }
elif m.status == GRB.INFEASIBLE:
    m.computeIIS()
    m.write("iis.ilp")
    _aoe_result = {"status": "infeasible", "objective_value": None, "variables": {}}
elif m.status == GRB.UNBOUNDED:
    _aoe_result = {"status": "unbounded", "objective_value": None, "variables": {}}
else:
    _aoe_result = {"status": f"other:{m.status}", "objective_value": None, "variables": {}}

# pathlib avoids calling the built-in open(), which a Gurobi variable named
# "open" (e.g. for facility-location problems) would shadow.
_aoe_pathlib.Path("result.json").write_text(_aoe_json.dumps(_aoe_result))
'''


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_user_message(state: GraphState) -> str:
    """Send only the structural model to the LLM — no numerical data."""
    return json.dumps(
        {"milp_model": state.get("milp_model", {})},
        ensure_ascii=False,
        indent=2,
    )


def _build_data_section(state: GraphState) -> str:
    """
    Build the data section as Python source code to prepend to the LLM output.

    Generates:
      - One list per set:       I = ["W1", "W2"]
      - One dict per parameter: c = {"W1": {"S1": 4.5, ...}}

    Sets come from milp_model["sets"][i]["elements"] (populated by input_retrieval).
    Parameters are mapped via milp_model["parameters"][i]["data_key"] -> raw_data[data_key],
    and written under the parameter's name (not data_key).
    """
    milp_model = state.get("milp_model", {})
    raw_data   = state.get("raw_data", {})
    lines      = ["# --- Data Section (auto-generated) ---"]

    for s in milp_model.get("sets", []):
        name     = s["name"]
        elements = s.get("elements", [])
        lines.append(f"{name} = {repr(elements)}")

    lines.append("")

    for p in milp_model.get("parameters", []):
        name     = p["name"]
        data_key = p.get("data_key", name)
        data     = raw_data.get(data_key, {})
        lines.append(f"{name} = {repr(data)}")

    lines.append("")
    return "\n".join(lines)


def _extract_code(text: str) -> str:
    """
    Pull the Python code block out of the LLM response.
    Accepts ```python ... ``` or plain ``` ... ```.
    Falls back to the raw response if no fence is found.
    """
    match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _check_syntax(code: str) -> str | None:
    """
    Check the generated script for syntax errors.
    Returns an error message string, or None if the code is valid.
    """
    try:
        ast.parse(code)
        return None
    except SyntaxError as e:
        return f"SyntaxError at line {e.lineno}: {e.msg}"


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def code_generator_node(state: GraphState) -> dict:
    """
    LangGraph node for the CodeGenerator agent.

    1. Sends only milp_model to the LLM.
    2. Extracts the Python code block from the response.
    3. Prepends the data section (sets + parameters from raw_data).
    4. Checks the final script for syntax errors.
    5. Returns generated_code (full script) and any syntax error found.
    """
    llm          = get_llm_client()
    user_content = _build_user_message(state)
    data_section = _build_data_section(state)
    last_code    = ""
    syntax_error = None

    for _ in range(1, _MAX_RETRIES + 1):
        response = llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ])

        # Track token usage and costs
        usage = response.response_metadata.get("token_usage", {})
        if usage:
            CostCalculator.update_state_costs(
                state=state,
                agent_name="code_generator",
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
            )

        model_code = _extract_code(response.content)
        if not model_code:
            continue

        full_code    = data_section + "\n" + model_code + "\n" + _RESULT_EPILOGUE
        syntax_error = _check_syntax(full_code)

        if syntax_error is None:
            last_code = full_code
            break
        # syntax error — retry with the same prompt
        last_code = full_code  # keep last attempt even if broken

    return {
        "generated_code": last_code,
        "code_syntax_error": syntax_error,
        "costs": state.get("costs", {"total": 0.0, "by_agent": {}}),
    }
