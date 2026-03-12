"""
agents/explainer.py — ExplainerAgent: interprets solver results in plain language.

Reads solver_result and milp_model from GraphState, calls GPT-4o to produce a
human-readable solution explanation, and generates a LaTeX formulation
stored in GraphState.latex_model.
"""
