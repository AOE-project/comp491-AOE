"""
middleware/handle.py — AOEHandle.

Entry point for all UIs. Runs the LangGraph workflow, returns Events,
and delegates session persistence to SessionManager.
Returns (updated_state, list[Event]) to the caller.
"""
