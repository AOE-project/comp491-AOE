"""
agents/debug.py — DebugAgent: corrects errors/feedbacks in Gurobi scripts.

Uses last_error or generated_script from GraphState to construct a repair prompt,
writes the corrected script back to state, and records each attempt in debug_attempts.
Raises DebugLimitError when the maximum number of retry attempts is exceeded.
"""
