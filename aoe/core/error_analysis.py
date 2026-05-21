"""
core/error_analysis.py — Rich error context capture and analysis.

Extracts traceback, identifies error type, suggests recovery strategies,
and formats strategic prompts for the LLM recovery agents.

Architecture:
1. Catch & Inspect: Extract traceback, state, original goal
2. Classification: Identify error category
3. Analysis: Suggest potential causes
4. Strategic Prompting: Structure LLM prompt with analysis hints
"""

import re
import keyword
from typing import Optional


class ErrorContext:
    """Rich error context with analysis."""
    
    def __init__(self, error_message: str, generated_code: str, milp_model: dict = None):
        self.error_message = error_message
        self.generated_code = generated_code
        self.milp_model = milp_model or {}
        self.error_type = self._classify_error()
        self.analysis = self._analyze_error()
    
    def _classify_error(self) -> str:
        """
        Classify error into 4 categories:
        1. "syntax_error": Python syntax parsing issues
        2. "runtime_error": Undefined variables/names
        3. "modeling_error": Gurobi API misuse
        4. "unknown_error": Other errors
        """
        msg = self.error_message.lower()
        
        # 1. Syntax errors - Python syntax parsing
        if any(x in msg for x in ["syntaxerror", "invalid syntax", "unexpected"]):
            return "syntax_error"
        
        # 2. Runtime errors - Undefined variables/names
        if any(x in msg for x in ["nameerror", "undefined", "not defined", "not found"]):
            return "runtime_error"
        
        # 3. Modeling errors - Gurobi API misuse
        if any(x in msg for x in ["gurobipy", "quicksum", "attr", "gurobi", "addvars", "addconstr"]):
            return "modeling_error"
        
        # 4. Unknown errors - everything else
        return "unknown_error"
    
    def _analyze_error(self) -> dict:
        """Analyze error and suggest potential causes for 4 error types."""
        analysis = {
            "error_type": self.error_type,
            "causes": [],
            "recovery_hints": []
        }

        if self.error_type == "syntax_error":
            # Check for reserved keyword usage (e.g., "lambda = 100")
            reserved_kw = None
            for kw in keyword.kwlist:
                if f"{kw} =" in self.generated_code or f"{kw}=" in self.generated_code:
                    reserved_kw = kw
                    break

            if reserved_kw:
                analysis["causes"] = [
                    f"Using Python reserved keyword '{reserved_kw}' as a variable name",
                    "Python does not allow reserved keywords to be assigned to",
                    "Common reserved keywords: lambda, class, def, return, if, else, for, while, import, etc."
                ]
                analysis["recovery_hints"] = [
                    f"Rename '{reserved_kw}' to something else (e.g., '{reserved_kw}_weight', '{reserved_kw}_param', 'penalty_{reserved_kw}')",
                    "Use alternative variable names like 'weight', 'penalty', 'coefficient', or 'param'",
                    "Avoid all Python reserved keywords when creating variables"
                ]
            else:
                analysis["causes"] = [
                    "Malformed Python syntax in generated code",
                    "Incorrect indentation or brackets",
                    "Invalid Gurobi method calls"
                ]
                analysis["recovery_hints"] = [
                    "Check Python syntax around the error line",
                    "Verify all brackets and parentheses are balanced",
                    "Ensure Gurobi API calls match function signatures"
                ]
        
        elif self.error_type == "runtime_error":
            # Extract variable name if possible
            var_match = re.search(r"name '(\w+)' is not defined", self.error_message)
            if var_match:
                var_name = var_match.group(1)
                analysis["causes"] = [
                    f"Variable '{var_name}' is used but never defined",
                    f"'{var_name}' should be defined in the data section",
                    f"'{var_name}' might be a parameter or set name from the model"
                ]
                analysis["recovery_hints"] = [
                    f"Add '{var_name}' to the data section before using it",
                    f"Check if '{var_name}' is spelled correctly (typo?)",
                    f"Verify '{var_name}' is in model['sets'] or model['parameters']"
                ]
            else:
                analysis["causes"] = [
                    "A variable/function is used but not defined",
                    "Typo in variable name",
                    "Missing import statement"
                ]
                analysis["recovery_hints"] = [
                    "Define all variables before using them",
                    "Check spelling of all variable names",
                    "Ensure all required imports are present"
                ]
        
        elif self.error_type == "modeling_error":
            analysis["causes"] = [
                "Incorrect Gurobi API usage (wrong method or parameters)",
                "Index mismatch when accessing sets or parameters",
                "Incorrect dimension handling in addVars or addConstrs"
            ]
            analysis["recovery_hints"] = [
                "Verify Gurobi method signatures (e.g., addVars syntax)",
                "Check if loop indices match parameter dimensions",
                "Ensure all variable/constraint definitions use correct Gurobi methods",
                "Review Gurobi documentation for tuplelist/tupledict operations"
            ]
        
        elif self.error_type == "unknown_error":
            analysis["causes"] = [
                "Unable to classify error from message",
                "Unexpected error type not in known categories",
                "Possible issue with code generation or runtime environment"
            ]
            analysis["recovery_hints"] = [
                "Review the full error message and traceback",
                "Check the generated code for obvious issues",
                "Verify that all imports and dependencies are available",
                "Consider manually reviewing the model structure"
            ]
        
        return analysis
    
    def to_llm_prompt(self, attempt_number: int = 1, previous_attempts: list = None) -> str:
        """
        Format a strategic prompt for the LLM recovery agent.

        Includes:
        - Role instruction
        - Error context with traceback
        - Generated code that failed
        - Analysis with hints
        - Previous attempts (if any)
        """
        previous_attempts = previous_attempts or []

        prompt = f"""You are a Gurobi Programming Expert fixing a Mixed-Integer Linear Programming (MILP) optimization script.

## Error Context (Attempt #{attempt_number})

**Error Classification:** {self.analysis['error_type'].upper()}
**Error Message:** {self.error_message}

## Potential Causes
{chr(10).join(f"- {cause}" for cause in self.analysis['causes'])}

## Recovery Hints (CRITICAL — Follow These)
{chr(10).join(f"- {hint}" for hint in self.analysis['recovery_hints'])}
"""

        # Add extra emphasis if this is a repeated error
        if attempt_number > 1:
            prompt += f"\n⚠️  **ATTENTION**: This is attempt #{attempt_number} and the SAME error is occurring repeatedly.\nYou MUST identify and fix the root cause, not generate the same broken code."

        prompt += f"""

## Generated Code (FAILED)
```python
{self.generated_code}
```

## MILP Model Structure (for reference)
```json
{self._format_model_structure()}
```

## Your Task
1. Analyze the error deeply
2. Identify the root cause from the list above
3. Provide ONLY the corrected Python code in a ```python code block
4. Do NOT include explanations outside the code block (we'll parse it automatically)
5. Ensure:
   - All variables used are defined
   - All set and parameter names match the model
   - Gurobi API calls are correct
   - Data dimensions match the model structure
   - NO Python reserved keywords are used as variable names

## Previous Attempts (if any)
{self._format_previous_attempts(previous_attempts)}

Now, provide the corrected code:
"""
        return prompt
    
    def _format_model_structure(self) -> str:
        """Format MILP model as JSON for LLM context."""
        import json
        try:
            # Only send structure, not full data
            model_structure = {
                "sets": [
                    {"name": s["name"], "elements": f"[{len(s.get('elements', []))} items]"}
                    for s in self.milp_model.get("sets", [])
                ],
                "parameters": [
                    {"name": p["name"], "data_key": p.get("data_key")}
                    for p in self.milp_model.get("parameters", [])
                ],
                "objective": self.milp_model.get("objective", {}),
                "constraints": self.milp_model.get("constraints", [])[:3] + (
                    [{"note": "..."}] if len(self.milp_model.get("constraints", [])) > 3 else []
                )
            }
            return json.dumps(model_structure, indent=2, ensure_ascii=False)
        except:
            return str(self.milp_model)
    
    def _format_previous_attempts(self, attempts: list) -> str:
        """Format previous failed attempts for LLM context."""
        if not attempts:
            return "None — this is the first attempt."

        lines = []
        for i, attempt in enumerate(attempts, 1):
            lines.append(f"\n### Attempt #{i}")
            lines.append(f"**Error:** {attempt.get('error', 'Unknown')}")
            error_type = attempt.get('error_type', 'unknown')
            analysis = attempt.get('analysis', {})
            causes = analysis.get('causes', [])
            if causes:
                lines.append(f"**Error Classification:** {error_type}")
                lines.append(f"**Identified issues:**")
                for cause in causes[:2]:  # Show top 2 causes
                    lines.append(f"  - {cause}")

        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        """Export as dictionary for state storage."""
        return {
            "error_message": self.error_message,
            "error_type": self.error_type,
            "analysis": self.analysis,
            "generated_code": self.generated_code,
        }
