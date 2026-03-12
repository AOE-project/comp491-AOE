"""
middleware/events.py — Event dataclasses emitted by AOEHandle.run().

Converts internal graph execution steps into a flat stream of events the UI and CLI
can consume: AgentStarted, AgentFinished, SolverExecuted, ErrorRaised, StateUpdated,
and UserInputNeeded etc.
"""
