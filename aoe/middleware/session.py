"""
middleware/session.py — SessionManager.

Creates, saves, and loads GraphState sessions by UUID.
Supports in-memory (dev) and file-system JSON (persistent) backends.
Raises SessionError for unknown or expired session IDs.
"""
