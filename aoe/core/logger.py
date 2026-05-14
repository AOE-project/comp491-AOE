"""
core/logger.py — Incremental session persistence.

Saves state and artefacts to sessions/<session_id>/ after every graph turn
so a session can be resumed from exactly where it left off.

Layout:
    sessions/<session_id>/
        metadata.json       created_at, last_updated, problem_description
        state.json          latest GraphState snapshot
        milp_model.json     written once the analyser populates it
        generated_code.py   output of CodeGenerator
        solver_result.csv   tabular solver output
        events.jsonl        append-only event log
        inputs/
            <data_key>.csv  one file per raw_data entry (costs, capacities, …)
"""

import csv
import json
import datetime
from pathlib import Path
from typing import Any


_DEFAULT_SESSIONS_ROOT = Path(__file__).resolve().parent.parent / "sessions"


class SessionLogger:

    def __init__(self, session_id: str, sessions_root: Path | None = None):
        self.session_id = session_id
        self._root = Path(sessions_root) if sessions_root else _DEFAULT_SESSIONS_ROOT
        self.dir = self._root / session_id
        self.dir.mkdir(parents=True, exist_ok=True)

    @property
    def _path_state(self) -> Path:
        return self.dir / "state.json"

    @property
    def _path_metadata(self) -> Path:
        return self.dir / "metadata.json"

    @property
    def _path_events(self) -> Path:
        return self.dir / "events.jsonl"

    @property
    def _path_inputs(self) -> Path:
        p = self.dir / "inputs"
        p.mkdir(exist_ok=True)
        return p

    def checkpoint(self, state: dict, label: str = "") -> None:
        """Save state snapshot and update any artefacts that are now populated."""
        now = _utcnow()

        self._path_state.write_text(
            json.dumps(_json_safe(state), indent=2, ensure_ascii=False), encoding="utf-8"
        )

        meta = self._load_metadata()
        if "created_at" not in meta:
            meta["created_at"] = now
            meta["session_id"] = state.get("session_id", self.session_id)
            meta["problem_description"] = state.get("problem_description", "")
        meta["last_updated"] = now
        meta["iteration_count"] = state.get("iteration_count", 0)
        meta["analyser_approved"] = state.get("analyser_approved", False)
        self._path_metadata.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

        if state.get("raw_data"):
            self.save_inputs(state["raw_data"])
        if state.get("milp_model"):
            self.save_milp_model(state["milp_model"])
        if state.get("generated_code"):
            self.save_generated_code(state["generated_code"])
        if state.get("solver_result"):
            self.save_solver_result(state["solver_result"])

        self.log_event("checkpoint", {
            "label": label,
            "iteration": state.get("iteration_count", 0),
            "analyser_approved": state.get("analyser_approved", False),
        })

    def log_event(self, event_type: str, data: dict | None = None) -> None:
        entry = {"ts": _utcnow(), "event": event_type, **(data or {})}
        with self._path_events.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def save_inputs(self, raw_data: dict) -> None:
        """Write each raw_data entry to inputs/<data_key>.csv.

        1-D param  {"row": value, ...}          → two-column CSV (row, value)
        2-D param  {"row": {"col": value, ...}} → matrix CSV with row/col headers
        """
        for key, value in raw_data.items():
            path = self._path_inputs / f"{key}.csv"
            with path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                if isinstance(value, dict) and value:
                    first = next(iter(value.values()))
                    if isinstance(first, dict):
                        # 2-D: rows × cols matrix
                        col_labels = list(first.keys())
                        writer.writerow([""] + col_labels)
                        for row_label, row_vals in value.items():
                            writer.writerow([row_label] + [row_vals.get(c, "") for c in col_labels])
                    else:
                        # 1-D: row → scalar
                        writer.writerow(["key", "value"])
                        for k, v in value.items():
                            writer.writerow([k, v])
                elif isinstance(value, list):
                    writer.writerow([key])
                    for item in value:
                        writer.writerow([item])
                else:
                    writer.writerow([key])
                    writer.writerow([value])

    def save_milp_model(self, model: dict) -> None:
        (self.dir / "milp_model.json").write_text(
            json.dumps(model, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def save_generated_code(self, code: str) -> None:
        (self.dir / "generated_code.py").write_text(code, encoding="utf-8")

    def save_solver_result(self, result: dict) -> None:
        path = self.dir / "solver_result.csv"
        variables = result.get("variables") or result.get("solution")

        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["status",          result.get("status", "")])
            writer.writerow(["objective_value", result.get("objective_value", "")])
            writer.writerow([])
            writer.writerow(["variable", "value"])
            if isinstance(variables, dict):
                for name, value in variables.items():
                    writer.writerow([name, value])
            elif isinstance(variables, list):
                for row in variables:
                    if isinstance(row, dict):
                        writer.writerow([row.get("name", ""), row.get("value", "")])
                    else:
                        writer.writerow(["", row])

    @classmethod
    def load(
        cls,
        session_id: str,
        sessions_root: Path | None = None,
    ) -> tuple["SessionLogger", dict]:
        """Load a saved session. Returns (logger, state)."""
        root = Path(sessions_root) if sessions_root else _DEFAULT_SESSIONS_ROOT
        session_dir = root / session_id
        if not session_dir.exists():
            raise FileNotFoundError(f"Session not found: {session_id!r}")

        state_path = session_dir / "state.json"
        if not state_path.exists():
            raise FileNotFoundError(f"No state.json found for session {session_id!r}")

        state = json.loads(state_path.read_text(encoding="utf-8"))
        logger = cls(session_id, sessions_root=root)
        logger.log_event("session_resumed", {"session_id": session_id})
        return logger, state

    @staticmethod
    def list_sessions(sessions_root: Path | None = None) -> list[dict]:
        """Return metadata for all saved sessions, newest first."""
        root = Path(sessions_root) if sessions_root else _DEFAULT_SESSIONS_ROOT
        if not root.exists():
            return []
        results = []
        for entry in root.iterdir():
            if not entry.is_dir():
                continue
            meta_path = entry / "metadata.json"
            if meta_path.exists():
                results.append(json.loads(meta_path.read_text(encoding="utf-8")))
        return sorted(results, key=lambda m: m.get("last_updated", ""), reverse=True)

    def _load_metadata(self) -> dict:
        if self._path_metadata.exists():
            return json.loads(self._path_metadata.read_text(encoding="utf-8"))
        return {}


def _utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(i) for i in obj]
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)
