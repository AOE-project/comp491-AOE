"""
agents/base.py — BaseAgent and PromptRegistry.

Shared abstract base for all four LLM agents. It initialises
an OpenAI client, tracks token usage and latency. It defines the
abstract run(state: GraphState) -> dict interface that each agent must implement.
"""
