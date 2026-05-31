"""End-to-end evaluator for the AOE pipeline.

Runs each benchmark problem through the full conversation loop:
- analyser question turns
- explicit approval step
- input retrieval turns (auto-answered)
- code generation + solver execution

This script reports pipeline-level metrics (not just analyser structure metrics).

Usage (from aoe/):
    python tests/evaluation/evaluate_end_to_end.py --input tests/evaluation/test.jsonl --limit 5 --verbose
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# End-to-end evaluation focuses on optimization pipeline completion.
# LaTeX rendering depends on external pdflatex; default to dummy for portability.
os.environ["USE_DUMMY_ANALYSER"] = "false"
os.environ["USE_DUMMY_CODE_GENERATOR"] = "false"
os.environ["TEST_ERROR_INJECTION"] = "false"
os.environ.setdefault("USE_DUMMY_LATEX_GENERATOR", "true")

from middleware.handle import AOEHandle  # noqa: E402


DEFAULT_INPUT_PATH = Path(__file__).resolve().with_name("test.jsonl")
MAX_TURNS_DEFAULT = 50


@dataclass
class AutoAnswerContext:
    numeric_pool: list[float]
    numeric_cursor: int = 0
    set_answers: dict[str, list[str]] = field(default_factory=dict)

    def next_number(self, fallback: float = 1.0) -> float:
        if self.numeric_cursor < len(self.numeric_pool):
            value = self.numeric_pool[self.numeric_cursor]
            self.numeric_cursor += 1
            return value
        return fallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate full AOE end-to-end pipeline")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Path to benchmark JSONL file")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N problems")
    parser.add_argument("--max-turns", type=int, default=MAX_TURNS_DEFAULT, help="Max interaction turns per problem")
    parser.add_argument("--verbose", action="store_true", help="Print per-problem details")
    return parser.parse_args()


def resolve_input_path(path: Path) -> Path:
    if path.is_absolute() and path.exists():
        return path

    candidates = [
        Path.cwd() / path,
        PROJECT_ROOT / path,
        PROJECT_ROOT.parent / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return path


def _unwrap_record(line_obj: dict[str, Any]) -> dict[str, Any]:
    if not line_obj:
        raise ValueError("Empty JSON object in dataset line")
    _, value = next(iter(line_obj.items()))
    if not isinstance(value, dict):
        raise ValueError("Record payload is not an object")
    return value


def load_records(path: Path) -> list[dict[str, Any]]:
    path = resolve_input_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            text = raw.strip()
            if not text:
                continue
            outer = json.loads(text)
            if not outer:
                raise ValueError("Empty JSON object in dataset line")
            problem_key, payload = next(iter(outer.items()))
            if not isinstance(payload, dict):
                raise ValueError("Record payload is not an object")
            rec = payload
            rec["_line_no"] = line_no
            rec["_problem_key"] = str(problem_key)
            records.append(rec)
    return records


def _to_float_if_possible(value: Any) -> float | None:
    s = str(value).strip().lower()
    word_map = {"third": 1.0 / 3.0, "half": 0.5, "twice": 2.0, "double": 2.0}
    if s in word_map:
        return word_map[s]
    try:
        return float(s)
    except ValueError:
        return None


def build_numeric_pool(record: dict[str, Any]) -> list[float]:
    pool: list[float] = []

    obj = record.get("obj_declaration", {})
    for v in (obj.get("terms", {}) or {}).values():
        num = _to_float_if_possible(v)
        if num is not None:
            pool.append(num)

    for c in record.get("const_declarations", []) or []:
        limit = _to_float_if_possible(c.get("limit"))
        if limit is not None:
            pool.append(limit)
        param = _to_float_if_possible(c.get("param"))
        if param is not None:
            pool.append(param)
        terms = c.get("terms", {}) or {}
        for v in terms.values():
            num = _to_float_if_possible(v)
            if num is not None:
                pool.append(num)

    for v in record.get("params", []) or []:
        num = _to_float_if_possible(v)
        if num is not None:
            pool.append(num)

    if not pool:
        pool = [1.0, 2.0, 3.0]
    return pool


def answer_open_questions() -> str:
    return (
        "Assume standard linear-programming conditions. "
        "All decision variables are non-negative unless otherwise stated. "
        "Use the natural interpretation of the problem statement."
    )


def answer_set_size(spec: dict[str, Any], record: dict[str, Any], ctx: AutoAnswerContext) -> str:
    set_name = str(spec.get("set_name", "S"))
    vars_ = [str(v) for v in (record.get("vars") or []) if str(v).strip()]

    if vars_:
        # Reuse human-readable labels for better downstream mapping.
        labels = [v.replace(" ", "_") for v in vars_]
    else:
        labels = [f"{set_name}1", f"{set_name}2"]

    ctx.set_answers[set_name] = labels
    return ", ".join(labels)


def answer_param_scalar(ctx: AutoAnswerContext) -> str:
    return str(ctx.next_number())


def answer_param_vector(spec: dict[str, Any], ctx: AutoAnswerContext) -> str:
    rows = spec.get("row_labels") or []
    values = [str(ctx.next_number()) for _ in rows]
    return "\n".join(values)


def answer_param_matrix(spec: dict[str, Any], ctx: AutoAnswerContext) -> str:
    rows = spec.get("row_labels") or []
    cols = spec.get("col_labels") or []
    lines: list[str] = []
    for _ in rows:
        row_values = [str(ctx.next_number()) for _ in cols]
        lines.append(",".join(row_values))
    return "\n".join(lines)


def answer_input_spec(spec: dict[str, Any], record: dict[str, Any], ctx: AutoAnswerContext) -> str:
    spec_type = spec.get("type")
    if spec_type == "set_size":
        return answer_set_size(spec, record, ctx)

    if spec_type == "param_data":
        shape = spec.get("shape") or []
        if not shape:
            return answer_param_scalar(ctx)

        col_labels = spec.get("col_labels")
        if col_labels is None:
            return answer_param_vector(spec, ctx)
        return answer_param_matrix(spec, ctx)

    return "1"


def run_end_to_end(record: dict[str, Any], max_turns: int, verbose: bool = False) -> dict[str, Any]:
    document = str(record.get("document", ""))
    handle = AOEHandle()
    ctx = AutoAnswerContext(numeric_pool=build_numeric_pool(record))

    turns = 0
    state = handle.run(document, None)

    while turns < max_turns:
        turns += 1

        # Finished: solver_result exists
        if state.get("solver_result"):
            break

        # Analyser clarification turns
        open_questions = state.get("open_questions") or []
        if open_questions:
            state = handle.run(answer_open_questions(), state)
            continue

        # Approval gate between analyser and input retrieval
        current_spec = state.get("current_input_spec") or {}
        if (
            state.get("iteration_count", 0) > 0
            and not state.get("analyser_approved", False)
            and not current_spec
            and not state.get("solver_result")
        ):
            state["analyser_approved"] = True
            state = handle.run("__approved__", state)
            continue

        # Input retrieval turns
        if current_spec.get("type"):
            answer = answer_input_spec(current_spec, record, ctx)
            state = handle.run(answer, state)
            continue

        # Stalled state fallback
        state = handle.run("continue", state)

    solver_result = state.get("solver_result") or {}
    status = str(solver_result.get("status", "missing"))
    completed = bool(solver_result)
    optimal = status == "optimal"

    if verbose:
        print(f"  turns={turns} completed={completed} status={status}")
        if solver_result.get("stderr"):
            print(f"  stderr={solver_result.get('stderr')}")

    return {
        "completed": completed,
        "optimal": optimal,
        "status": status,
        "turns": turns,
        "session_id": state.get("session_id"),
        "debug_count": len(state.get("debug_attempts") or []),
        "objective_value": solver_result.get("objective_value"),
        "variables": solver_result.get("variables") or {},
        "solver_result": solver_result,
    }


def evaluate(
    records: list[dict[str, Any]],
    max_turns: int,
    verbose: bool = False,
) -> dict[str, Any]:
    total = len(records)
    execution_ok_count = 0
    success_count = 0
    completion_count = 0
    status_hist: dict[str, int] = {}
    details: list[dict[str, Any]] = []

    for i, rec in enumerate(records, start=1):
        if verbose:
            print(f"[#{i}] line={rec.get('_line_no')}")

        result = run_end_to_end(rec, max_turns=max_turns, verbose=verbose)
        status = result["status"]
        completion = result["completed"]
        execution_ok = result["completed"] and status not in {"runtime_error", "timeout", "missing"}
        success = status == "optimal"

        completion_count += 1 if completion else 0
        execution_ok_count += 1 if execution_ok else 0
        success_count += 1 if success else 0
        status_hist[status] = status_hist.get(status, 0) + 1

        details.append(
            {
                "index": i,
                "line_no": rec.get("_line_no"),
                "completed": completion,
                "execution_ok": execution_ok,
                "success": success,
                "optimal": result["optimal"],
                "status": status,
                "turns": result["turns"],
                "debug_count": result["debug_count"],
                "objective_value": result["objective_value"],
                "variables": result["variables"],
                "session_id": result["session_id"],
            }
        )

    execution_rate = (execution_ok_count / total * 100.0) if total else 0.0
    success_rate = (success_count / total * 100.0) if total else 0.0
    completion_rate = (completion_count / total * 100.0) if total else 0.0

    return {
        "total": total,
        "completion_count": completion_count,
        "execution_ok_count": execution_ok_count,
        "success_count": success_count,
        "execution_rate": execution_rate,
        "success_rate": success_rate,
        "completion_rate": completion_rate,
        "status_hist": status_hist,
        "details": details,
    }


def print_report(metrics: dict[str, Any]) -> None:
    total = metrics["total"]
    completion = metrics["completion_count"]
    execution_ok = metrics["execution_ok_count"]
    success = metrics["success_count"]

    print()
    print(f"Total Problems: {total}")
    print()
    print(f"End-to-End Completion Rate: {metrics['completion_rate']:.1f}% ({completion}/{total})")
    print()
    print(f"Execution Rate: {metrics['execution_rate']:.1f}% ({execution_ok}/{total})")
    print()
    print(f"Success Rate: {metrics['success_rate']:.1f}% ({success}/{total})")

    print()
    print("Per-Problem Outcomes:")
    for row in metrics["details"]:
        objective_value = row.get("objective_value")
        objective_display = "None" if objective_value is None else f"{objective_value}"
        variables = row.get("variables") or {}
        var_items = sorted(variables.items(), key=lambda x: x[0])
        vars_display = ", ".join(f"{k}={v}" for k, v in var_items) if var_items else "-"
        print(
            f"- #{row['index']} (line {row['line_no']}): "
            f"status={row['status']}, objective={objective_display}, "
            f"debug_invocations={row['debug_count']}, vars={vars_display}"
        )

    total_debug_invocations = sum(int(row.get("debug_count", 0)) for row in metrics["details"])
    print()
    print(f"Total Debug Agent Invocations: {total_debug_invocations}")


def main() -> int:
    args = parse_args()

    try:
        records = load_records(args.input)
    except Exception as exc:
        print(f"Failed to load dataset: {exc}")
        return 1

    if not records:
        print("No records found in dataset.")
        return 1

    if args.limit is not None:
        if args.limit <= 0:
            print("--limit must be a positive integer.")
            return 1
        records = records[: args.limit]

    metrics = evaluate(
        records,
        max_turns=args.max_turns,
        verbose=args.verbose,
    )
    print_report(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
