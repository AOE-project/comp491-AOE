"""
ui/cli/main.py — Typer CLI (CLIApp) and Rich terminal formatters.

CLIApp defines a run command that starts an interactive session, passing user
messages to AOEHandle.run() and streaming events to the terminal. Supports
--session-id for session resume.
"""
