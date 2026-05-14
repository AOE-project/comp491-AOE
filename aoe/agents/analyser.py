"""
agents/analyser.py — AnalyserAgent.

Extracts a structural MILP formulation from the user via multi-turn dialogue.
Each call is one turn: it takes the current GraphState, calls the LLM, validates the output and returns an updated state dict.

The loop is done by AOEHandle: this node runs once per user message.
When open_questions is empty the agent is ready for user approval; the graph then waits for an explicit approval signal before advancing to the next node.
"""

import json
import re
from pathlib import Path

import jsonschema
import jsonschema.validators
from referencing import Registry, Resource
from langchain_core.messages import HumanMessage, SystemMessage

from core.config import get_llm_client
from core.state import GraphState
from core.cost_calculator import CostCalculator

# Paths

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
_SCHEMAS_DIR = Path(__file__).parent.parent / "schemas"

_SYSTEM_PROMPT = (_PROMPTS_DIR / "analyser_prompt.txt").read_text(encoding="utf-8")
_ANALYSER_OUTPUT_SCHEMA = json.loads((_SCHEMAS_DIR / "analyser_output.json").read_text(encoding="utf-8"))
_MILP_MODEL_SCHEMA = json.loads((_SCHEMAS_DIR / "milp_model.json").read_text(encoding="utf-8"))

_REGISTRY = Registry().with_resources([
    ("milp_model.json", Resource.from_contents(_MILP_MODEL_SCHEMA)),
])

_MAX_ITERATIONS = 10
_MAX_RETRIES = 3

# Helpers

def _extract_json(text: str) -> dict:
    """Extract and parse the first JSON object from an LLM response string."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM response.")
    return json.loads(match.group())


def _validate(output: dict) -> None:
    """Validate the full analyser output against its schema (resolves $ref locally)."""
    validator_cls = jsonschema.validators.validator_for(_ANALYSER_OUTPUT_SCHEMA)
    validator = validator_cls(_ANALYSER_OUTPUT_SCHEMA, registry=_REGISTRY)
    validator.validate(output)


def _build_user_message(state: GraphState) -> str:
    """Construct the user-turn content sent to the LLM."""
    problem_description = state.get("problem_description", "")
    previous_output = state.get("analyser_output")
    user_msg = state.get("history", [{}])[-1].get("content", "") if state.get("history") else ""

    user_answers = None
    if previous_output:
        previous_questions = previous_output.get("open_questions", [])
        if previous_questions:
            user_answers = {
                "previous_questions": previous_questions,
                "user_response": user_msg,
            }

    payload = {
        "problem_description": problem_description,
        "previous_output": previous_output,
        "user_answers": user_answers,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


# Node

def analyser_node(state: GraphState) -> dict:
    """
    LangGraph node for the Analyser agent.

    Calls the LLM, retries up to _MAX_RETRIES times on parse/validation failure,
    and returns a partial GraphState dict with updated analyser fields.
    """
    # Model already approved — pass through without touching state so the
    # router can advance to input_retrieval (or beyond).
    if state.get("analyser_approved"):
        return {}

    llm = get_llm_client()
    user_content = _build_user_message(state)

    last_error = None
    output = None

    if state.get("iteration_count", 0) >= _MAX_ITERATIONS:
        return {
            "open_questions": ["Maximum number of dialogue turns reached. Please start a new session."],
            "iteration_count": _MAX_ITERATIONS,
        }

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = llm.invoke([
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=user_content),
            ])
            raw = response.content
            output = _extract_json(raw)
            _validate(output)

            # Track token usage and costs
            usage = response.response_metadata.get("token_usage", {})
            if usage:
                CostCalculator.update_state_costs(
                    state=state,
                    agent_name="analyser",
                    input_tokens=usage.get("prompt_tokens", 0),
                    output_tokens=usage.get("completion_tokens", 0),
                )
            break
        except (ValueError, json.JSONDecodeError, jsonschema.ValidationError) as exc:
            last_error = exc
            if attempt == _MAX_RETRIES:
                return {
                    "open_questions": [
                        "I was unable to parse your input into a valid model structure after "
                        f"{_MAX_RETRIES} attempts. Could you rephrase or clarify your problem "
                        "description? Error: " + str(last_error)
                    ],
                    "iteration_count": state.get("iteration_count", 0) + 1,
                }

    milp_model = output.get("milp_model", {})

    return {
        "analyser_output": output,
        "milp_model": milp_model,
        "confirmed_assumptions": output.get("confirmed_assumptions", []),
        "unconfirmed_assumptions": output.get("unconfirmed_assumptions", []),
        "open_questions": output.get("open_questions", []),
        "analysis_summary": output.get("analysis_summary", ""),
        "technical_summary": output.get("technical_summary", ""),
        "iteration_count": state.get("iteration_count", 0) + 1,
        "analyser_approved": False,
        "costs": state.get("costs", {"total": 0.0, "by_agent": {}}),
    }
