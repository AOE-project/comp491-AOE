"""
tests/test_llm_connection.py — Integration test: verifies the OpenAI API key is valid
and the model responds using LangChain message types. Hits the real API; requires .env.

Run with:
    python3 -m pytest tests/test_llm_connection.py -v -s
"""

from langchain_core.messages import HumanMessage, SystemMessage

from core.config import get_llm_client


def test_llm_connection():
    llm = get_llm_client()

    messages = [
        SystemMessage(content="Translate the user message to French."),
        HumanMessage(content="Hi, this is Automated Optimization Engineer. How can I help?"),
    ]

    response = llm.invoke(messages)

    assert response.content and len(response.content.strip()) > 0, "Expected a non-empty response"
    print(f"\nModel reply: {response.content}")