"""
agents/input_retrieval.py — InputRetrievalNode.

Collects set elements and parameter data from the user, one item per graph turn.

Queue order: all sets first (to resolve shapes), then all parameters.

Each task in the queue is a dict:
    {
        "type": "set_size",
        "set_name": "W",
        "description": "Warehouses",
    }
    or
    {
        "type": "param_data",
        "param_name": "ship_cost",
        "data_key": "ship_cost",
        "shape": ["W", "S"],
        "description": "Shipping cost per unit from each warehouse to each store",
    }

current_input_spec written to state on each turn:
    {
        "type": "set_size" | "param_data",
        "prompt": str,
        "set_name": str,          # set_size only
        "param_name": str,        # param_data only
        "data_key": str,          # param_data only
        "shape": list[str],       # param_data only — resolved to element lists
        "row_labels": list[str],  # param_data, 1-D or 2-D (index-0 dimension)
        "col_labels": list[str],  # param_data, 2-D only (index-1 dimension)
        "error": str | None,      # set when last answer failed validation
    }

raw_data format:
    1-D param (shape [W]):   {"W1": 10.0, "W2": 20.0}
    2-D param (shape [W,S]): {"W1": {"S1": 4.5, "S2": 3.2}, "W2": {...}}
"""

import io
import csv
from pathlib import Path

import pandas as pd

from core.state import GraphState


# Queue builder

def _build_queue(milp_model: dict) -> list:
    """Return an ordered list of collection tasks from the MILP model."""
    queue = []

    for s in milp_model.get("sets", []):
        queue.append({
            "type": "set_size",
            "set_name": s["name"],
            "description": s.get("description", s["name"]),
        })

    for p in milp_model.get("parameters", []):
        queue.append({
            "type": "param_data",
            "param_name": p["name"],
            "data_key": p.get("data_key", p["name"]),
            "shape": p.get("shape", []),
            "description": p.get("description", p["name"]),
        })

    return queue


# Set element helpers

def _parse_set_answer(answer: str, set_name: str) -> tuple[list[str] | None, str | None]:
    """
    Parse the user's answer for a set_size task.

    Accepts:
      - A positive integer  → auto-generate labels  e.g. "3" → ["W1","W2","W3"]
      - Comma-separated names → use as-is           e.g. "WH1, WH2, WH3"

    Returns (elements, error).
    """
    answer = answer.strip()
    if not answer:
        return None, "Please enter a number or a comma-separated list of names."

    # Try integer first
    try:
        count = int(answer)
        if count <= 0:
            return None, "Please enter a positive integer."
        return [f"{set_name}{i + 1}" for i in range(count)], None
    except ValueError:
        pass

    # Treat as comma-separated names
    parts = [p.strip() for p in answer.split(",") if p.strip()]
    if not parts:
        return None, "Could not parse your answer. Enter a count or comma-separated names."
    return parts, None


def _apply_set_elements(state: GraphState, set_name: str, elements: list[str]) -> dict:
    """Return a state patch that writes elements into milp_model.sets[i]."""
    milp_model = {**state["milp_model"]}
    sets = [dict(s) for s in milp_model.get("sets", [])]
    for s in sets:
        if s["name"] == set_name:
            s["elements"] = elements
            break
    milp_model["sets"] = sets
    return {"milp_model": milp_model}


def _get_elements(state: GraphState, set_name: str) -> list[str]:
    """Return the resolved elements for a set (must already be collected)."""
    for s in state["milp_model"].get("sets", []):
        if s["name"] == set_name:
            return s.get("elements", [])
    return []


# Spec builder helpers

def _set_size_prompt(task: dict) -> str:
    sn = task["set_name"]
    desc = task["description"]
    return (
        f"How many {desc} ({sn}) are there? "
        f"Enter a count (e.g. 3) or list element names comma-separated "
        f"(e.g. {sn}1, {sn}2, {sn}3)."
    )


def _param_data_prompt(task: dict, row_labels: list[str], col_labels: list[str] | None) -> str:
    pname = task["param_name"]
    desc = task["description"]
    shape = task["shape"]

    if not shape:
        # Scalar
        return (
            f"Enter the value for **{pname}** ({desc}).\n"
            f"Type a single number."
        )
    elif col_labels is None:
        # 1-D
        return (
            f"Provide values for **{pname}** ({desc}).\n"
            f"Shape: [{' × '.join(shape)}]  —  {len(row_labels)} value(s).\n"
            f"Upload a CSV file or, in the CLI, enter the file path."
        )
    else:
        # 2-D
        return (
            f"Provide values for **{pname}** ({desc}).\n"
            f"Shape: [{' × '.join(shape)}]  —  "
            f"{len(row_labels)} row(s) × {len(col_labels)} column(s).\n"
            f"Upload a CSV file (rows = {shape[0]}, columns = {shape[1]}) "
            f"or, in the CLI, enter the file path."
        )


def _make_spec(task: dict, state: GraphState, error: str | None = None) -> dict:
    if task["type"] == "set_size":
        return {
            "type": "set_size",
            "set_name": task["set_name"],
            "description": task["description"],
            "prompt": _set_size_prompt(task),
            "error": error,
        }

    # param_data — resolve labels from already-collected set elements
    shape = task["shape"]
    row_labels = _get_elements(state, shape[0]) if shape else []
    col_labels = _get_elements(state, shape[1]) if len(shape) > 1 else None

    return {
        "type": "param_data",
        "param_name": task["param_name"],
        "data_key": task["data_key"],
        "shape": shape,
        "is_scalar": not shape,
        "row_labels": row_labels,
        "col_labels": col_labels,
        "prompt": _param_data_prompt(task, row_labels, col_labels),
        "error": error,
    }


# Parameter data parsing

def _load_dataframe(source: str) -> tuple[pd.DataFrame | None, str | None]:
    """
    Load a DataFrame from a file path or raw CSV text.
    Returns (df, error).
    """
    source = source.strip()

    # Try as file path first
    path = Path(source)
    if path.exists() and path.is_file():
        try:
            return pd.read_csv(path, header=None), None
        except Exception as e:
            return None, f"Could not read file '{path}': {e}"

    # Try as inline CSV text
    try:
        reader = csv.reader(io.StringIO(source))
        rows = [row for row in reader if any(cell.strip() for cell in row)]
        if rows:
            return pd.DataFrame(rows), None
    except Exception:
        pass

    return None, (
        f"Could not find file '{source}'. "
        "Please provide a valid file path or paste CSV values directly."
    )


def _parse_param_answer(
    answer: str,
    row_labels: list[str],
    col_labels: list[str] | None,
) -> tuple[dict | None, str | None]:
    """
    Parse user answer for a param_data task.

    1-D (col_labels is None):  expects one value per row_label.
    2-D:                       expects len(row_labels) rows × len(col_labels) columns.

    Returns (data_dict, error).
    """
    df, err = _load_dataframe(answer)
    if err:
        return None, err

    # Strip any header row/column that looks like labels
    # We always trust shape over content: take first len(row_labels) rows,
    # first (1 or len(col_labels)) columns.
    try:
        values = df.values
    except Exception as e:
        return None, f"Could not read data: {e}"

    def _is_numeric(v) -> bool:
        s = str(v).strip().lower()
        if s in ("nan", "inf", "-inf", ""):
            return False
        try:
            float(s)
            return True
        except ValueError:
            return False

    if col_labels is None:
        # 1-D: expect a single value per row_label.
        # The Gradio table prepends a row-label column, so we may receive
        # extra columns/rows that must be stripped before extracting numbers.
        rows_arr = values.tolist()

        # Skip first row if it contains no numeric values (column header row)
        if rows_arr and not any(_is_numeric(c) for c in rows_arr[0]):
            rows_arr = rows_arr[1:]

        # Skip first column of each row if it is non-numeric (row label column)
        if rows_arr and not _is_numeric(rows_arr[0][0]):
            rows_arr = [r[1:] for r in rows_arr]

        numeric_flat = []
        for row in rows_arr:
            for cell in row:
                try:
                    numeric_flat.append(float(str(cell).strip()))
                except ValueError:
                    return None, f"Non-numeric value found: '{cell}'"

        if len(numeric_flat) != len(row_labels):
            return None, (
                f"Expected {len(row_labels)} value(s) for {row_labels}, "
                f"got {len(numeric_flat)}."
            )

        return {label: numeric_flat[i] for i, label in enumerate(row_labels)}, None

    else:
        # 2-D: expect len(row_labels) rows × len(col_labels) cols.
        rows_arr = values.tolist()

        # Skip first row if it contains no numeric values (column header row)
        if rows_arr and not any(_is_numeric(c) for c in rows_arr[0]):
            rows_arr = rows_arr[1:]

        # Skip first column of each row if it's non-numeric (row label column)
        if rows_arr and not _is_numeric(rows_arr[0][0]):
            rows_arr = [r[1:] for r in rows_arr]

        if len(rows_arr) != len(row_labels):
            return None, (
                f"Expected {len(row_labels)} row(s) ({row_labels}), "
                f"got {len(rows_arr)}."
            )

        result = {}
        for i, rl in enumerate(row_labels):
            row = rows_arr[i]
            if len(row) != len(col_labels):
                return None, (
                    f"Row '{rl}': expected {len(col_labels)} value(s) "
                    f"({col_labels}), got {len(row)}."
                )
            inner = {}
            for j, cl in enumerate(col_labels):
                try:
                    inner[cl] = float(str(row[j]).strip())
                except ValueError:
                    return None, f"Non-numeric value at ({rl}, {cl}): '{row[j]}'"
            result[rl] = inner

        return result, None


# Main node

def input_retrieval_node(state: GraphState) -> dict:
    """
    LangGraph node — collects set elements and parameter data one turn at a time.

    First call (queue empty): builds queue, sets current_input_spec for the
    first item, returns immediately (the user's "__approved__" message is ignored).

    Subsequent calls: parses the last user message as the answer to current_input_spec,
    validates, stores result, advances cursor, sets spec for next item (or clears it
    when all items are collected).
    """
    queue: list = list(state.get("input_retrieval_queue") or [])
    cursor: int = state.get("input_retrieval_cursor", 0)

    # First call: build the queue
    if not queue:
        queue = _build_queue(state["milp_model"])
        if not queue:
            # Nothing to collect (no sets, no parameters) — pass through
            return {
                "input_retrieval_queue": queue,
                "input_retrieval_cursor": 0,
                "current_input_spec": {},
            }
        spec = _make_spec(queue[0], state)
        
        return {
            "input_retrieval_queue": queue,
            "input_retrieval_cursor": 0,
            "current_input_spec": spec,
        }

    # Already complete (e.g. session resumed after all data collected)
    if cursor >= len(queue):
        return {"current_input_spec": {}}

    # Subsequent calls: process the user's answer
    history = state.get("history", [])
    # Last user message is the answer to the current prompt
    user_messages = [m for m in history if m.get("role") == "user"]
    answer = user_messages[-1]["content"] if user_messages else ""

    current_task = queue[cursor]

    # set_size task
    if current_task["type"] == "set_size":
        elements, err = _parse_set_answer(answer, current_task["set_name"])
        if err:
            spec = _make_spec(current_task, state, error=err)
            return {"current_input_spec": spec}

        patch = _apply_set_elements(state, current_task["set_name"], elements)
        cursor += 1

        if cursor >= len(queue):
            patch.update({
                "input_retrieval_queue": queue,
                "input_retrieval_cursor": cursor,
                "current_input_spec": {},
            })
            return patch

        # Merge milp_model patch into state temporarily so _make_spec can
        # resolve set elements for any upcoming param tasks
        merged_state = {**state, **patch}
        next_spec = _make_spec(queue[cursor], merged_state)
        patch.update({
            "input_retrieval_queue": queue,
            "input_retrieval_cursor": cursor,
            "current_input_spec": next_spec,
        })
        return patch

    # param_data task
    if current_task["type"] == "param_data":
        shape = current_task["shape"]

        # Scalar parameter (shape=[]) — expect a single number typed in chat
        if not shape:
            try:
                data = float(answer.strip())
            except ValueError:
                spec = _make_spec(current_task, state, error="Please enter a single numeric value.")
                return {"current_input_spec": spec}
        else:
            row_labels = _get_elements(state, shape[0])
            col_labels = _get_elements(state, shape[1]) if len(shape) > 1 else None
            data, err = _parse_param_answer(answer, row_labels, col_labels)
            if err:
                spec = _make_spec(current_task, state, error=err)
                return {"current_input_spec": spec}

        raw_data = {**state.get("raw_data", {}), current_task["data_key"]: data}
        cursor += 1

        if cursor >= len(queue):
            return {
                "raw_data": raw_data,
                "input_retrieval_queue": queue,
                "input_retrieval_cursor": cursor,
                "current_input_spec": {},
            }

        next_spec = _make_spec(queue[cursor], state)
        return {
            "raw_data": raw_data,
            "input_retrieval_queue": queue,
            "input_retrieval_cursor": cursor,
            "current_input_spec": next_spec,
        }

    # Unknown task type — skip
    cursor += 1
    return {"input_retrieval_cursor": cursor}
