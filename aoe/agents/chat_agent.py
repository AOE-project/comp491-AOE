"""
agents/chat_agent.py — ChatAgent.

Inputs  (from GraphState): milp_model, solver_result, chat_history, history,
                           chat_pending_modification
Output  (to  GraphState):  chat_response, chat_history,
                           chat_pending_modification, chat_modification_approved

Post-solver interactive agent with four intents:
  - explain              → answers the user's question; no model change
  - propose_modification → describes the change and asks for yes/no confirmation;
                           stores original user request in chat_pending_modification
  - confirm_modification → user said yes; sets chat_modification_approved=True so
                           AOEHandle re-runs the analyser (with the user's original
                           request) followed by the full pipeline
  - reject_modification  → user said no; clears chat_pending_modification
"""

import json
import re
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from core.config import get_llm_client
from core.state import GraphState

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

    Classifies the user's message and signals AOEHandle. The chat agent never
    modifies milp_model directly — modifications are applied by the analyser.
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
                }

    intent = output.get("intent", "explain")
    response_text = output.get("response", "")

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
    }

    if intent == "propose_modification":
        # Store the original user request so AOEHandle can forward it to the analyser.
        updates["chat_pending_modification"] = {
            "description": response_text,
            "user_request": current_user_message,
            "set_changes": bool(output.get("set_changes", False)),
            "regen_force_ask": output.get("regen_force_ask") or [],
        }

    elif intent == "confirm_modification":
        # Signal AOEHandle to re-run analyser + pipeline with the original request.
        updates["chat_modification_approved"] = True
        updates["chat_pending_modification"] = None

    elif intent == "reject_modification":
        updates["chat_pending_modification"] = None

    return updates
