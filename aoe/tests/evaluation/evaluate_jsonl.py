"""Evaluate LLM extraction quality on a JSONL benchmark.

This script reads problems from a JSONL file, sends each problem to the project's
Analyser pipeline, and compares the output against reference annotations.

Primary metrics:
- Execution Rate: LLM call + parsing completed without runtime failure
- Average Modelling Accuracy: Weighted objective, variable, and constraint quality
- Error Distribution: constraint count, operator, and objective errors

Usage (from aoe/ directory):
    python tests/evaluation/evaluate_jsonl.py --input tests/evaluation/test.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


# Ensure imports like "from agents.analyser import analyser_node" work when run as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.analyser import analyser_node  # noqa: E402


DEFAULT_INPUT_PATH = Path(__file__).resolve().with_name("test.jsonl")
DEFAULT_AOE_INPUT_PATH = Path(__file__).resolve().with_name("aoe_test.jsonl")
DEFAULT_AOE_SCHEMA_PATH = Path(__file__).resolve().with_name("aoe_eval_schema.json")
OBJECTIVE_MODES = ("strict", "relaxed")
EVAL_MODES = ("nl4opt", "aoe")
PAIR_MATCH_THRESHOLD = 0.25


OP_TO_SYMBOL = {
    "LESS_OR_EQUAL": "<=",
    "GREATER_OR_EQUAL": ">=",
    "EQUAL": "=",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate analyser output on JSONL benchmark")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to benchmark JSONL file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-problem diagnostics",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N problems",
    )
    parser.add_argument(
        "--objective-mode",
        choices=OBJECTIVE_MODES,
        default="relaxed",
        help="Objective comparison strictness",
    )
    parser.add_argument(
        "--mode",
        choices=EVAL_MODES,
        default="nl4opt",
        help="Evaluation mode: nl4opt-style or aoe semantic mode",
    )
    parser.add_argument(
        "--aoe-input",
        type=Path,
        default=DEFAULT_AOE_INPUT_PATH,
        help="Path to AOE benchmark JSONL file (used when --mode aoe)",
    )
    parser.add_argument(
        "--aoe-schema",
        type=Path,
        default=DEFAULT_AOE_SCHEMA_PATH,
        help="Path to AOE evaluation schema JSON file (used when --mode aoe)",
    )
    return parser.parse_args()


def _unwrap_record(line_obj: dict[str, Any]) -> dict[str, Any]:
    """Dataset lines are shaped as {"<id>": {...record...}}; unwrap them."""
    if not line_obj:
        raise ValueError("Empty JSON object in dataset line")
    _, value = next(iter(line_obj.items()))
    if not isinstance(value, dict):
        raise ValueError("Record payload is not an object")
    return value


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
            try:
                outer = json.loads(text)
                rec = _unwrap_record(outer)
                rec["_line_no"] = line_no
                records.append(rec)
            except Exception as exc:
                raise ValueError(f"Failed to parse line {line_no}: {exc}") from exc
    return records


def load_plain_jsonl(path: Path) -> list[dict[str, Any]]:
    path = resolve_input_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            text = raw.strip()
            if not text:
                continue
            rec = json.loads(text)
            if not isinstance(rec, dict):
                raise ValueError(f"Line {line_no} is not a JSON object")
            rec["_line_no"] = line_no
            records.append(rec)
    return records


def load_json_file(path: Path) -> dict[str, Any]:
    path = resolve_input_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def make_state(problem: str) -> dict[str, Any]:
    return {
        "session_id": "evaluation-session",
        "problem_description": problem,
        "history": [{"role": "user", "content": problem}],
        "iteration_count": 0,
        "analyser_output": None,
        "milp_model": {},
        "confirmed_assumptions": [],
        "unconfirmed_assumptions": [],
        "open_questions": [],
        "analysis_summary": "",
        "technical_summary": "",
        "analyser_approved": False,
        "raw_data": {},
        "generated_code": "",
        "solver_result": {},
        "explanation": "",
        "token_usage": {},
    }


def run_model(document: str) -> tuple[bool, dict[str, Any], str | None]:
    state = make_state(document)
    try:
        result = analyser_node(state)
        milp_model = result.get("milp_model", {}) if isinstance(result, dict) else {}
        is_ok = isinstance(milp_model, dict) and bool(milp_model)
        return is_ok, milp_model, None
    except Exception as exc:  # pragma: no cover - runtime safety path
        return False, {}, str(exc)


def _normalize_num_str(s: str) -> str:
    # Convert numerically equivalent strings (e.g., "50", "50.0") to canonical form.
    try:
        val = float(s)
        if val.is_integer():
            return str(int(val))
        return str(val)
    except ValueError:
        return s.strip()


def _tokenize_keywords(*parts: str) -> list[str]:
    tokens: list[str] = []
    for part in parts:
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_]*", part.lower()):
            if len(token) >= 3:
                tokens.append(token)
    return tokens


def _canonical_var_name(var_expr: str) -> str:
    return var_expr.split("[", 1)[0].strip()


def _token_set(*parts: str) -> set[str]:
    return set(_tokenize_keywords(*parts))


def _overlap_score(left: set[str], right: set[str]) -> float:
    if not left:
        return 0.0
    return len(left & right) / len(left)


def _match_ratio(canonical_items: list[str], model_surface: str, threshold: float = 0.3) -> float:
    if not canonical_items:
        return 1.0
    surface_tokens = _token_set(model_surface)
    hits = 0
    for item in canonical_items:
        item_tokens = _token_set(item)
        if _overlap_score(item_tokens, surface_tokens) >= threshold:
            hits += 1
    return hits / len(canonical_items)


def evaluate_aoe(records: list[dict[str, Any]], schema: dict[str, Any], verbose: bool = False) -> dict[str, Any]:
    total = len(records)
    execution_ok = 0
    details: list[dict[str, Any]] = []

    mode_cfg = schema.get("modes", {}).get("aoe", {})
    w_instance = float(mode_cfg.get("weights", {}).get("instance_match", 0.2))
    w_alg = float(mode_cfg.get("weights", {}).get("algebraic_fidelity", 0.8))
    comp = mode_cfg.get("algebraic_components", {})
    w_sets = float(comp.get("sets", 0.2))
    w_params = float(comp.get("parameters", 0.2))
    w_vars = float(comp.get("variables", 0.3))
    w_cons = float(comp.get("semantic_constraints", 0.3))
    thr_cfg = mode_cfg.get("thresholds", {})
    token_thr = float(thr_cfg.get("token_overlap_match", 0.3))
    sem_thr = float(thr_cfg.get("semantic_constraint_match", 0.35))

    aoe_score_total = 0.0
    algebraic_total = 0.0
    instance_total = 0.0

    for i, rec in enumerate(records, start=1):
        document = str(rec.get("document", ""))
        aoe_ref = rec.get("aoe_reference", {}) or {}
        nl4_ref = rec.get("nl4opt_reference", {}) or {}

        ran_ok, milp_model, run_err = run_model(document)
        if ran_ok:
            execution_ok += 1

        pred_obj = milp_model.get("objective", {}) if isinstance(milp_model, dict) else {}
        pred_constraints = milp_model.get("constraints", []) if isinstance(milp_model, dict) else []
        pred_sets = milp_model.get("sets", []) if isinstance(milp_model, dict) else []
        pred_params = milp_model.get("parameters", []) if isinstance(milp_model, dict) else []
        pred_vars = milp_model.get("variables", []) if isinstance(milp_model, dict) else []

        if not isinstance(pred_constraints, list):
            pred_constraints = []
        if not isinstance(pred_sets, list):
            pred_sets = []
        if not isinstance(pred_params, list):
            pred_params = []
        if not isinstance(pred_vars, list):
            pred_vars = []

        # Instance-style subscore (low weight): direction + approximate constraint count
        exp_direction = str(nl4_ref.get("objective_direction", "")).strip().lower()
        pred_direction = str(pred_obj.get("sense", "")).strip().lower()
        dir_ok = 1.0 if (exp_direction and exp_direction == pred_direction) else 0.0
        exp_count = int(nl4_ref.get("constraint_count", 0) or 0)
        pred_count = len(pred_constraints)
        if exp_count <= 0:
            count_score = 1.0
        else:
            count_score = max(0.0, 1.0 - abs(pred_count - exp_count) / exp_count)
        instance_match = 0.5 * dir_ok + 0.5 * count_score

        set_surface = " ".join(str(x.get("name", "")) + " " + str(x.get("description", "")) for x in pred_sets if isinstance(x, dict))
        param_surface = " ".join(str(x.get("name", "")) + " " + str(x.get("description", "")) for x in pred_params if isinstance(x, dict))
        var_surface = " ".join(str(x.get("name", "")) + " " + str(x.get("description", "")) for x in pred_vars if isinstance(x, dict))
        cons_surface = "\n".join(
            (str(c.get("name", "")) + " " + str(c.get("description", "")) + " " + str(c.get("expression", "")))
            for c in pred_constraints if isinstance(c, dict)
        )

        canonical_sets = [str(x) for x in (aoe_ref.get("canonical_sets") or [])]
        canonical_params = [str(x) for x in (aoe_ref.get("canonical_params") or [])]
        canonical_vars = [_canonical_var_name(str(x)) for x in (aoe_ref.get("canonical_vars") or [])]
        semantic_constraints = [str(x.get("logic", "")) for x in (aoe_ref.get("semantic_constraints") or []) if isinstance(x, dict)]

        set_score = _match_ratio(canonical_sets, set_surface, threshold=token_thr)
        param_score = _match_ratio(canonical_params, param_surface, threshold=token_thr)
        var_score = _match_ratio(canonical_vars, var_surface + "\n" + cons_surface, threshold=token_thr)

        if semantic_constraints:
            matched = 0
            pred_constraint_texts = [
                str(c.get("name", "")) + " " + str(c.get("description", "")) + " " + str(c.get("expression", ""))
                for c in pred_constraints if isinstance(c, dict)
            ]
            for logic in semantic_constraints:
                logic_tokens = _token_set(logic)
                best = 0.0
                for text in pred_constraint_texts:
                    best = max(best, _overlap_score(logic_tokens, _token_set(text)))
                if best >= sem_thr:
                    matched += 1
            sem_constraint_score = matched / len(semantic_constraints)
        else:
            sem_constraint_score = 1.0

        algebraic_fidelity = (
            w_sets * set_score
            + w_params * param_score
            + w_vars * var_score
            + w_cons * sem_constraint_score
        )

        aoe_score = w_instance * instance_match + w_alg * algebraic_fidelity

        aoe_score_total += aoe_score
        algebraic_total += algebraic_fidelity
        instance_total += instance_match

        row = {
            "index": i,
            "line_no": rec.get("_line_no"),
            "problem_id": rec.get("problem_id", f"line_{rec.get('_line_no', i)}"),
            "execution_ok": ran_ok,
            "instance_match": instance_match,
            "algebraic_fidelity": algebraic_fidelity,
            "aoe_score": aoe_score,
            "set_score": set_score,
            "param_score": param_score,
            "var_score": var_score,
            "semantic_constraint_score": sem_constraint_score,
            "error": run_err,
        }
        details.append(row)

        if verbose:
            print(
                f"[#{i}] exec={ran_ok} instance={instance_match:.2f} algebraic={algebraic_fidelity:.2f} aoe={aoe_score:.2f} "
                f"sets={set_score:.2f} params={param_score:.2f} vars={var_score:.2f} sem_cons={sem_constraint_score:.2f}"
            )
            if run_err:
                print(f"     error: {run_err}")

    return {
        "mode": "aoe",
        "total": total,
        "execution_ok": execution_ok,
        "execution_rate": (execution_ok / total * 100.0) if total else 0.0,
        "average_instance_match": (instance_total / total * 100.0) if total else 0.0,
        "average_algebraic_fidelity": (algebraic_total / total * 100.0) if total else 0.0,
        "average_aoe_score": (aoe_score_total / total * 100.0) if total else 0.0,
        "details": details,
    }


def _expected_constraint_tokens(expected: dict[str, Any]) -> set[str]:
    constraint_type = str(expected.get("type", "")).strip().lower()
    parts = [str(expected.get("direction", ""))]

    if constraint_type in {"lowerbound", "upperbound"}:
        parts.append(str(expected.get("var", "")))
    elif constraint_type in {"xy", "xby"}:
        parts.extend([str(expected.get("x_var", "")), str(expected.get("y_var", "")), str(expected.get("param", ""))])
    elif constraint_type == "linear":
        # Linear constraints in the dataset often represent semantic resources such as budget/capital/space.
        parts.append(str(expected.get("type", "")))
    elif constraint_type == "sum":
        parts.extend([str(expected.get("type", "")), str(expected.get("direction", ""))])

    if not any(parts):
        parts.extend(str(key) for key in expected.get("terms", {}).keys())
    return _token_set(*parts)


def _predicted_constraint_text(predicted: dict[str, Any]) -> str:
    return " ".join(
        [
            str(predicted.get("name", "")),
            str(predicted.get("description", "")),
            str(predicted.get("expression", "")),
        ]
    )


def _predicted_constraint_tokens(predicted: dict[str, Any]) -> set[str]:
    return _token_set(_predicted_constraint_text(predicted))


def summarize_expected_constraint(expected: dict[str, Any]) -> str:
    parts = [str(expected.get("type", "")).strip()]
    operator = normalize_expected_operator(str(expected.get("operator", "")))
    if operator:
        parts.append(operator)

    for key in ("var", "x_var", "y_var", "limit", "param"):
        value = str(expected.get(key, "")).strip()
        if value:
            parts.append(f"{key}={value}")

    terms = expected.get("terms", {})
    if isinstance(terms, dict) and terms:
        parts.append("terms=" + ", ".join(f"{k}:{v}" for k, v in terms.items()))

    return " | ".join(parts)


def summarize_predicted_constraint(predicted: dict[str, Any]) -> str:
    name = str(predicted.get("name", "")).strip() or "unnamed"
    expr = str(predicted.get("expression", "")).strip() or "missing"
    return f"{name} | {expr}"


def summarize_expected_objective(expected_obj: dict[str, Any]) -> str:
    sense = str(expected_obj.get("direction", "")).strip() or "missing"
    name = str(expected_obj.get("name", "")).strip() or "missing"
    terms = expected_obj.get("terms", {})
    term_text = ", ".join(f"{k}:{v}" for k, v in terms.items()) if isinstance(terms, dict) and terms else "none"
    return f"sense={sense} | name={name} | terms={term_text}"


def summarize_predicted_objective(pred_obj: dict[str, Any]) -> str:
    sense = str(pred_obj.get("sense", "")).strip() or "missing"
    expr = str(pred_obj.get("expression", "")).strip() or "missing"
    return f"sense={sense} | expression={expr}"


def _is_tolerated_extra_constraint(predicted: dict[str, Any]) -> bool:
    text = _predicted_constraint_text(predicted).lower()
    if "nonnegative" in text or "non-neg" in text:
        return True
    if re.search(r">=\s*0(?:\D|$)", text):
        return True
    if re.search(r"0\s*<=", text):
        return True
    if extract_operator_symbol(str(predicted.get("expression", ""))) == "=":
        return True
    return False


def _model_surface_text(milp_model: dict[str, Any]) -> str:
    parts: list[str] = []
    for collection_name in ("sets", "parameters", "variables", "constraints"):
        collection = milp_model.get(collection_name, [])
        if not isinstance(collection, list):
            continue
        for item in collection:
            if not isinstance(item, dict):
                continue
            parts.extend(
                [
                    str(item.get("name", "")),
                    str(item.get("description", "")),
                    str(item.get("expression", "")),
                ]
            )
    objective = milp_model.get("objective", {})
    if isinstance(objective, dict):
        parts.extend([str(objective.get("sense", "")), str(objective.get("expression", ""))])
    return " ".join(parts)


def _best_variable_overlap(expected_var: str, model_surface_text: str) -> float:
    expected_tokens = _token_set(expected_var)
    predicted_tokens = _token_set(model_surface_text)
    return _overlap_score(expected_tokens, predicted_tokens)


def compare_variables(expected_vars: list[str], milp_model: dict[str, Any]) -> tuple[float, int, list[str]]:
    if not expected_vars:
        return 1.0, 0, []

    model_surface_text = _model_surface_text(milp_model)
    matched = 0
    reasons: list[str] = []
    for expected_var in expected_vars:
        overlap = _best_variable_overlap(expected_var, model_surface_text)
        if overlap >= PAIR_MATCH_THRESHOLD:
            matched += 1
        else:
            reasons.append(f"Variable Missing: {expected_var}")

    score = matched / len(expected_vars)
    variable_errors = len(expected_vars) - matched
    return score, variable_errors, reasons


def objective_match_strict(expected_obj: dict[str, Any], pred_obj: dict[str, Any]) -> bool:
    expected_sense = str(expected_obj.get("direction", "")).strip().lower()
    pred_sense = str(pred_obj.get("sense", "")).strip().lower()
    if expected_sense != pred_sense:
        return False

    expected_terms = expected_obj.get("terms", {})
    expected_coeffs = [_normalize_num_str(str(v)) for v in expected_terms.values()]

    pred_expr = str(pred_obj.get("expression", ""))
    # Extract scalar numbers from objective expression.
    found_nums = re.findall(r"(?<![A-Za-z_])[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", pred_expr)
    found_coeffs = [_normalize_num_str(x) for x in found_nums]

    expected_counter = Counter(expected_coeffs)
    found_counter = Counter(found_coeffs)
    # All expected coefficients must appear at least as many times in prediction.
    return all(found_counter[k] >= v for k, v in expected_counter.items())


def objective_match_relaxed(expected_obj: dict[str, Any], pred_obj: dict[str, Any]) -> bool:
    expected_sense = str(expected_obj.get("direction", "")).strip().lower()
    pred_sense = str(pred_obj.get("sense", "")).strip().lower()
    if expected_sense != pred_sense:
        return False

    pred_expr = str(pred_obj.get("expression", "")).strip()
    if not pred_expr:
        return False

    keyword_parts = [str(expected_obj.get("name", ""))]
    keyword_parts.extend(str(key) for key in expected_obj.get("terms", {}).keys())
    keywords = _tokenize_keywords(*keyword_parts)
    if not keywords:
        return True

    pred_expr_lower = pred_expr.lower()
    return any(keyword in pred_expr_lower for keyword in keywords) or bool(re.search(r"sum|profit|cost|score|revenue", pred_expr_lower))


def objective_match(expected_obj: dict[str, Any], pred_obj: dict[str, Any], mode: str) -> bool:
    if mode == "strict":
        return objective_match_strict(expected_obj, pred_obj)
    return objective_match_relaxed(expected_obj, pred_obj)


def extract_operator_symbol(expr: str) -> str | None:
    # Order matters: check <=/>= before < and >.
    if "<=" in expr:
        return "<="
    if ">=" in expr:
        return ">="
    if "==" in expr:
        return "="

    # A single '=' in math-style constraints is treated as equality.
    # Exclude obvious arrows and assignment-like tokens minimally.
    if "=" in expr and "=>" not in expr and "<=" not in expr and ">=" not in expr:
        return "="

    if "<" in expr:
        return "<"
    if ">" in expr:
        return ">"
    return None


def normalize_expected_operator(operator: str) -> str | None:
    normalized = operator.strip().upper()
    if normalized in OP_TO_SYMBOL:
        return OP_TO_SYMBOL[normalized]
    if normalized in {"<=", ">=", "="}:
        return normalized
    return None


def compare_constraints(
    expected_constraints: list[dict[str, Any]],
    predicted_constraints: list[dict[str, Any]],
) -> tuple[bool, int, int, float, int, list[str]]:
    expected_count = len(expected_constraints)
    predicted_count = len(predicted_constraints)
    reasons: list[str] = []

    if not expected_constraints:
        tolerated = sum(1 for pred in predicted_constraints if _is_tolerated_extra_constraint(pred))
        bad_extras = len(predicted_constraints) - tolerated
        if bad_extras:
            reasons.append(f"Unexpected Constraints: {bad_extras}")
        return bad_extras == 0, 0 if bad_extras == 0 else 1, 0, 1.0, tolerated, reasons

    pair_candidates: list[tuple[float, int, int]] = []
    for exp_idx, expected in enumerate(expected_constraints):
        exp_tokens = _expected_constraint_tokens(expected)
        for pred_idx, predicted in enumerate(predicted_constraints):
            pred_tokens = _predicted_constraint_tokens(predicted)
            overlap = _overlap_score(exp_tokens, pred_tokens)
            if overlap >= PAIR_MATCH_THRESHOLD:
                pair_candidates.append((overlap, exp_idx, pred_idx))

    pair_candidates.sort(reverse=True)
    matched_expected: dict[int, int] = {}
    matched_predicted: set[int] = set()
    for _, exp_idx, pred_idx in pair_candidates:
        if exp_idx in matched_expected or pred_idx in matched_predicted:
            continue
        matched_expected[exp_idx] = pred_idx
        matched_predicted.add(pred_idx)

    operator_errors = 0
    for exp_idx, pred_idx in sorted(matched_expected.items()):
        exp_op = str(expected_constraints[exp_idx].get("operator", ""))
        exp_symbol = normalize_expected_operator(exp_op)
        pred_expr = str(predicted_constraints[pred_idx].get("expression", ""))
        pred_symbol = extract_operator_symbol(pred_expr)
        if exp_symbol is None or pred_symbol is None or exp_symbol != pred_symbol:
            operator_errors += 1
            reasons.append(
                f"Operator Mismatch for matched constraint {exp_idx + 1}: Expected {exp_symbol or exp_op}, Got {pred_symbol or 'unknown'}"
            )

    unmatched_expected = [idx for idx in range(expected_count) if idx not in matched_expected]
    if unmatched_expected:
        reasons.append(f"Missing Expected Constraints: {len(unmatched_expected)}")

    unmatched_predicted = [idx for idx in range(predicted_count) if idx not in matched_predicted]
    tolerated_extras = 0
    bad_extras = 0
    for pred_idx in unmatched_predicted:
        predicted = predicted_constraints[pred_idx]
        if _is_tolerated_extra_constraint(predicted):
            tolerated_extras += 1
        else:
            bad_extras += 1

    if bad_extras:
        reasons.append(f"Unexpected Extra Constraints: {bad_extras}")
    if tolerated_extras:
        reasons.append(f"Tolerated Extra Non-Negativity Constraints: {tolerated_extras}")

    count_error = 0 if not unmatched_expected and bad_extras == 0 else 1
    matched_count = len(matched_expected)
    constraint_score = matched_count / expected_count
    constraints_ok = matched_count == expected_count and operator_errors == 0 and bad_extras == 0
    return constraints_ok, count_error, operator_errors, constraint_score, tolerated_extras, reasons


def get_failure_reasons(
    expected_obj: dict[str, Any],
    pred_obj: dict[str, Any],
    objective_ok: bool,
    variable_reasons: list[str],
    constraint_reasons: list[str],
) -> list[str]:
    reasons: list[str] = []

    if not objective_ok:
        expected_sense = str(expected_obj.get("direction", "")).strip().lower() or "missing"
        pred_sense = str(pred_obj.get("sense", "")).strip().lower() or "missing"
        pred_expr = str(pred_obj.get("expression", "")).strip() or "missing"
        reasons.append(f"Objective Mismatch: Sense {expected_sense} vs {pred_sense}; Expression {pred_expr}")

    reasons.extend(variable_reasons)
    reasons.extend(constraint_reasons)

    return reasons


def evaluate(records: list[dict[str, Any]], verbose: bool = False, objective_mode: str = "relaxed") -> dict[str, Any]:
    total = len(records)
    execution_ok = 0

    objective_errors = 0
    variable_errors = 0
    constraint_count_errors = 0
    wrong_operator_errors = 0
    tolerated_extra_constraints = 0
    modelling_accuracy_total = 0.0

    details: list[dict[str, Any]] = []

    for i, rec in enumerate(records, start=1):
        document = str(rec.get("document", ""))
        expected_obj = rec.get("obj_declaration", {})
        expected_vars = rec.get("vars", [])
        expected_constraints = rec.get("const_declarations", [])

        ran_ok, milp_model, run_err = run_model(document)
        if ran_ok:
            execution_ok += 1

        pred_obj = milp_model.get("objective", {}) if isinstance(milp_model, dict) else {}
        pred_variables = milp_model.get("variables", []) if isinstance(milp_model, dict) else []
        pred_constraints = milp_model.get("constraints", []) if isinstance(milp_model, dict) else []
        if not isinstance(pred_variables, list):
            pred_variables = []
        if not isinstance(pred_constraints, list):
            pred_constraints = []

        obj_ok = ran_ok and objective_match(expected_obj, pred_obj, objective_mode)
        objective_score = 1.0 if obj_ok else 0.0
        variable_score, var_err, variable_reasons = compare_variables(expected_vars, milp_model if isinstance(milp_model, dict) else {})
        cons_ok, count_err, op_err, constraint_score, tolerated_count, constraint_reasons = compare_constraints(expected_constraints, pred_constraints)
        modelling_accuracy = 0.3 * objective_score + 0.4 * variable_score + 0.3 * constraint_score

        objective_errors += 0 if obj_ok else 1
        variable_errors += var_err
        constraint_count_errors += count_err
        wrong_operator_errors += op_err
        tolerated_extra_constraints += tolerated_count
        modelling_accuracy_total += modelling_accuracy

        failure_reasons = get_failure_reasons(
            expected_obj,
            pred_obj,
            obj_ok,
            variable_reasons,
            constraint_reasons,
        )

        row = {
            "index": i,
            "line_no": rec.get("_line_no"),
            "execution_ok": ran_ok,
            "objective_ok": obj_ok,
            "variable_score": variable_score,
            "constraints_ok": cons_ok,
            "constraint_score": constraint_score,
            "modelling_accuracy": modelling_accuracy,
            "constraint_count_error": bool(count_err),
            "operator_errors": op_err,
            "tolerated_extra_constraints": tolerated_count,
            "failure_reasons": failure_reasons,
            "error": run_err,
        }
        details.append(row)

        if verbose:
            print(
                f"[#{i}] exec={ran_ok} obj={obj_ok} cons={cons_ok} "
                f"count_err={bool(count_err)} op_err={op_err} "
                f"var_score={variable_score:.2f} cons_score={constraint_score:.2f} model_score={modelling_accuracy:.2f}"
            )
            print(f"     expected objective:  {summarize_expected_objective(expected_obj)}")
            print(f"     predicted objective: {summarize_predicted_objective(pred_obj)}")
            print(f"     expected vars ({len(expected_vars)}):  {', '.join(expected_vars) if expected_vars else 'none'}")
            predicted_var_names = [str(var.get('name', '')).strip() for var in pred_variables if isinstance(var, dict)]
            print(
                f"     predicted vars ({len(predicted_var_names)}): "
                f"{', '.join(name for name in predicted_var_names if name) if predicted_var_names else 'none'}"
            )
            print(f"     expected constraints ({len(expected_constraints)}):")
            for expected_constraint in expected_constraints:
                print(f"       - {summarize_expected_constraint(expected_constraint)}")
            print(f"     predicted constraints ({len(pred_constraints)}):")
            for predicted_constraint in pred_constraints:
                print(f"       - {summarize_predicted_constraint(predicted_constraint)}")
            if run_err:
                print(f"     error: {run_err}")
            elif failure_reasons:
                print(f"     reasons: {' | '.join(failure_reasons)}")

    exec_rate = (execution_ok / total * 100.0) if total else 0.0
    average_modelling_accuracy = (modelling_accuracy_total / total * 100.0) if total else 0.0

    return {
        "total": total,
        "execution_ok": execution_ok,
        "execution_rate": exec_rate,
        "objective_errors": objective_errors,
        "variable_errors": variable_errors,
        "constraint_count_errors": constraint_count_errors,
        "wrong_operator_errors": wrong_operator_errors,
        "tolerated_extra_constraints": tolerated_extra_constraints,
        "average_modelling_accuracy": average_modelling_accuracy,
        "details": details,
        "objective_mode": objective_mode,
    }


def print_report(metrics: dict[str, Any]) -> None:
    total = metrics["total"]
    execution_ok = metrics["execution_ok"]

    print()
    print(f"Total Problems: {total}")
    print()
    print(f"Objective Matching Mode: {metrics['objective_mode']}")
    print()
    print(
        "Executable Outputs (Execution Rate): "
        f"{metrics['execution_rate']:.1f}% ({execution_ok}/{total})"
    )
    print()
    print(
        "Average Modelling Accuracy: "
        f"{metrics['average_modelling_accuracy']:.1f}%"
    )
    print()
    print(
        "Error Distribution: "
        f"{metrics['objective_errors']} objective errors, "
        f"{metrics['variable_errors']} variable misses, "
        f"{metrics['constraint_count_errors']} constraint count errors, "
        f"{metrics['wrong_operator_errors']} wrong operator usages, "
        f"{metrics['tolerated_extra_constraints']} tolerated extra non-negativity constraints."
    )


def print_report_aoe(metrics: dict[str, Any]) -> None:
    total = metrics["total"]
    execution_ok = metrics["execution_ok"]

    print()
    print(f"Total Problems: {total}")
    print()
    print(
        "Executable Outputs (Execution Rate): "
        f"{metrics['execution_rate']:.1f}% ({execution_ok}/{total})"
    )
    print()
    print(f"Average Instance Match: {metrics['average_instance_match']:.1f}%")
    print(f"Average Algebraic Fidelity: {metrics['average_algebraic_fidelity']:.1f}%")
    print(f"Average AOE Score: {metrics['average_aoe_score']:.1f}%")


def main() -> int:
    args = parse_args()

    if args.mode == "aoe":
        try:
            records = load_plain_jsonl(args.aoe_input)
            schema = load_json_file(args.aoe_schema)
        except Exception as exc:
            print(f"Failed to load AOE inputs: {exc}")
            return 1

        if not records:
            print("No records found in AOE dataset.")
            return 1

        if args.limit is not None:
            if args.limit <= 0:
                print("--limit must be a positive integer.")
                return 1
            records = records[: args.limit]

        metrics = evaluate_aoe(records, schema=schema, verbose=args.verbose)
        print_report_aoe(metrics)
        return 0

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

    metrics = evaluate(records, verbose=args.verbose, objective_mode=args.objective_mode)
    print_report(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
