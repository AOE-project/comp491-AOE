"""
agents/chat_agent.py — ChatAgent.

Inputs  (from GraphState): milp_model, solver_result, chat_history, history
Output  (to  GraphState):  chat_response, chat_history, chat_modification_approved

Post-solver interactive agent with two intents:
  - explain  → answers the user's question in plain language, no model change
  - modify   → signals that the user wants a model change; sets
               chat_modification_approved=True so AOEHandle re-runs the
               analyser (with the user's request as feedback) followed by
               code_generator → solver → explainer.
"""

import json
import re
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from core.config import get_llm_client
from core.state import GraphState
from core.cost_calculator import CostCalculator

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
_SYSTEM_PROMPT = (_PROMPTS_DIR / "chat_agent_prompt.txt").read_text(encoding="utf-8")

_MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM response.")
    return json.loads(match.group())


def _build_user_message(state: GraphState) -> str:
    current_user_message = (
        state["history"][-1]["content"] if state.get("history") else ""
    )
    payload = {
        "milp_model": state.get("milp_model", {}),
        "solver_result": state.get("solver_result", {}),
        "chat_history": state.get("chat_history", []),
        "current_user_message": current_user_message,
        "pending_modification": state.get("chat_pending_modification"),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def chat_agent_node(state: GraphState) -> dict:
    """
    LangGraph node for the Chat Agent.

    Classifies the user's message as 'explain' or 'modify'.
    For 'modify', sets chat_modification_approved=True so AOEHandle
    can re-run the analyser with the user's feedback and re-optimise.
    """
    llm = get_llm_client()
    user_content = _build_user_message(state)

    output = None
    last_error = None

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = llm.invoke([
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=user_content),
            ])
            
            # Track token usage and costs
            usage = response.response_metadata.get("token_usage", {})

            if usage:
                CostCalculator.update_state_costs(
                    state=state,
                    agent_name="chat_agent",
                    input_tokens=usage.get("prompt_tokens", 0),
                    output_tokens=usage.get("completion_tokens", 0),
                )

            output = _extract_json(response.content)
            if "intent" not in output or "response" not in output:
                raise ValueError("Missing required fields.")
            break
        except Exception as exc:
            last_error = exc
            if attempt == _MAX_RETRIES:
                return {
                    "chat_response": (
                        "I had trouble processing your request. "
                        f"Please try again. ({last_error})"
                    ),
                    "costs": state.get("costs", {"total": 0.0, "by_agent": {}}),
                }

    intent = output.get("intent", "explain")
    response_text = output.get("response", "")
    updated_model = output.get("updated_milp_model")

    current_user_message = (
        state["history"][-1]["content"] if state.get("history") else ""
    )
    new_chat_history = state.get("chat_history", []) + [
        {"role": "user", "content": current_user_message},
        {"role": "assistant", "content": response_text},
    ]

    updates: dict = {
        "chat_response": response_text,
        "chat_history": new_chat_history,
        "chat_modification_approved": False,
        "costs": state.get("costs", {"total": 0.0, "by_agent": {}}),
    }

    if intent == "propose_modification" and updated_model:
        # Store the proposed change and wait for user confirmation.
        updates["chat_pending_modification"] = {
            "description": response_text,
            "updated_milp_model": updated_model,
        }

    elif intent == "confirm_modification":
        # User said yes — apply the pending (or LLM-returned) model.
        pending = state.get("chat_pending_modification") or {}
        approved_model = pending.get("updated_milp_model") or updated_model
        if approved_model:
            updates["milp_model"] = approved_model
            updates["chat_modification_approved"] = True
            updates["chat_pending_modification"] = None

    elif intent == "reject_modification":
        updates["chat_pending_modification"] = None

    return updates
