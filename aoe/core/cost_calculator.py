"""
core/cost_calculator.py — CostCalculator for tracking API costs per agent.

Tracks input/output token costs for gpt-5.4-nano.
Costs are stored in state['costs'] as: {'total': float, 'by_agent': {agent_name: float}}
"""

from typing import Dict


class CostCalculator:
    """Calculate and track API costs for LLM calls."""

    # Hardcoded pricing for gpt-5.4-nano (per 1M tokens)
    PRICING = {
        "gpt-5.4-nano": {
            "input": 0.075,      # $0.075 per 1M input tokens
            "output": 0.30,      # $0.30 per 1M output tokens
        }
    }

    @staticmethod
    def calculate_call_cost(
        input_tokens: int,
        output_tokens: int,
        model: str = "gpt-5.4-nano",
    ) -> float:
        """Calculate cost for a single API call in USD.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: Model name (default: gpt-5.4-nano)

        Returns:
            Cost in USD (float)
        """
        if model not in CostCalculator.PRICING:
            return 0.0

        pricing = CostCalculator.PRICING[model]
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost

    @staticmethod
    def update_state_costs(
        state: dict,
        agent_name: str,
        input_tokens: int,
        output_tokens: int,
        model: str = "gpt-5.4-nano",
    ) -> None:
        """Update state with new API costs in USD.

        Args:
            state: GraphState dict
            agent_name: Name of the agent making the call (e.g., 'analyser', 'code_generator')
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: Model name
        """
        cost = CostCalculator.calculate_call_cost(input_tokens, output_tokens, model)

        # Initialize costs structure if not present
        if "costs" not in state:
            state["costs"] = {"total": 0.0, "by_agent": {}}

        # Update total cost
        state["costs"]["total"] += cost

        # Update per-agent cost
        if agent_name not in state["costs"]["by_agent"]:
            state["costs"]["by_agent"][agent_name] = 0.0
        state["costs"]["by_agent"][agent_name] += cost

    @staticmethod
    def get_costs_summary(state: dict) -> Dict:
        """Get current costs from state.

        Returns:
            Dict with 'total' and 'by_agent' keys
        """
        return state.get("costs", {"total": 0.0, "by_agent": {}})
