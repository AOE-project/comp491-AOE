"""
agents/analyser.py — Analyser.

Extracts a structural MILP formulation from the user via multi-turn dialogue.
Emits a validated milp_model dict or sets status to "awaiting_user" if more input is needed.
"""
