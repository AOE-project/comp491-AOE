"""
agents/latex_generator.py — LaTeXGeneratorAgent.

Converts the finalised milp_model (from GraphState) into a compiled PNG by:
  1. Calling the LLM to produce a LaTeX document body.
  2. Wrapping the body in a minimal article template.
  3. Compiling with pdflatex.
  4. Converting the PDF to a PNG via PyMuPDF (no poppler required).

Runs once, immediately after analyser approval, before data collection begins.
Pass-through if latex_model is already populated.
PNG is saved to sessions/<session_id>/milp_formulation.png.
"""

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import fitz  # pymupdf — no poppler dependency

from langchain_core.messages import HumanMessage, SystemMessage

from core.config import get_llm_client
from core.state import GraphState
from core.cost_calculator import CostCalculator


_PROMPTS_DIR  = Path(__file__).parent.parent / "prompts"
_SESSIONS_DIR = Path(__file__).parent.parent / "sessions"

_TEX_SEARCH_PATHS = [
    "/Library/TeX/texbin",          # BasicTeX / MacTeX on macOS
    "/usr/local/texlive/2024/bin/universal-darwin",
    "/usr/local/texlive/2023/bin/universal-darwin",
    "/usr/local/bin",
    "/usr/bin",
]


def _find_pdflatex() -> str:
    """Return the absolute path to pdflatex, or raise FileNotFoundError."""
    for directory in _TEX_SEARCH_PATHS:
        candidate = Path(directory) / "pdflatex"
        if candidate.exists():
            return str(candidate)
    found = shutil.which("pdflatex")
    if found:
        return found
    raise FileNotFoundError(
        "pdflatex not found. Install BasicTeX with:\n"
        "  brew install --cask basictex\n"
        "then open a new terminal so /Library/TeX/texbin is on PATH."
    )

_SYSTEM_PROMPT = (_PROMPTS_DIR / "latex_generator_prompt.txt").read_text(encoding="utf-8")

_MAX_RETRIES = 3

_DOC_TEMPLATE = r"""\documentclass[11pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage[paperwidth=17cm, margin=1.5cm]{geometry}
\usepackage{parskip}
\pagestyle{empty}
\begin{document}
%s
\end{document}
"""


# Helpers

def _compile_to_png(latex_body: str, session_id: str) -> str:
    """Compile a LaTeX body to PNG, save under sessions/<session_id>/, return path."""
    pdflatex = _find_pdflatex()

    out_dir = _SESSIONS_DIR / session_id
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "milp_formulation.png"

    with tempfile.TemporaryDirectory() as tmp:
        tex_file = Path(tmp) / "model.tex"
        tex_file.write_text(_DOC_TEMPLATE % latex_body, encoding="utf-8")

        tex_bin = str(Path(pdflatex).parent)
        env = {**os.environ, "PATH": f"{tex_bin}:{os.environ.get('PATH', '')}"}
        result = subprocess.run(
            [pdflatex, "-interaction=nonstopmode", "model.tex"],
            cwd=tmp,
            capture_output=True,
            text=True,
            env=env,
        )
        pdf_file = Path(tmp) / "model.pdf"
        if not pdf_file.exists():
            raise RuntimeError(f"pdflatex failed:\n{result.stdout[-800:]}")

        from PIL import Image, ImageChops

        doc = fitz.open(str(pdf_file))
        mat = fitz.Matrix(150 / 72, 150 / 72)
        pages = [
            Image.frombytes("RGB", (p.width, p.height), p.samples)
            for p in (page.get_pixmap(matrix=mat) for page in doc)
        ]
        doc.close()

        white = (255, 255, 255)
        cropped = []
        for img in pages:
            bbox = ImageChops.difference(img, Image.new("RGB", img.size, white)).getbbox()
            bottom = min(img.height, bbox[3] + 20) if bbox else img.height
            cropped.append(img.crop((0, 0, img.width, bottom)))

        combined = Image.new("RGB", (cropped[0].width, sum(p.height for p in cropped)), white)
        y = 0
        for page in cropped:
            combined.paste(page, (0, y))
            y += page.height
        combined.save(str(png_path), "PNG")

    return str(png_path)


# Node

def latex_generator_node(state: GraphState) -> dict:
    """
    LangGraph node for the LaTeX Generator agent.

    Calls the LLM to produce a LaTeX body, compiles it to PNG via pdflatex,
    and returns {"latex_model": <body>, "latex_png_path": <path>}.
    Pass-through (returns {}) if latex_model is already populated.
    """
    if state.get("latex_model"):
        return {}

    milp_model   = state.get("milp_model") or {}
    user_content = json.dumps(milp_model, ensure_ascii=False, indent=2)
    session_id   = state.get("session_id", "unknown")

    llm        = get_llm_client()
    last_error = None

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = llm.invoke([
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=user_content),
            ])

            # Track token usage and costs
            usage = response.response_metadata.get("token_usage", {})
            if usage:
                CostCalculator.update_state_costs(
                    state=state,
                    agent_name="latex_generator",
                    input_tokens=usage.get("prompt_tokens", 0),
                    output_tokens=usage.get("completion_tokens", 0),
                )

            latex_body = response.content.strip()
            if not latex_body:
                raise ValueError("LLM returned an empty response.")

            png_path = _compile_to_png(latex_body, session_id)
            return {
                "latex_model": latex_body,
                "latex_png_path": png_path,
                "costs": state.get("costs", {"total": 0.0, "by_agent": {}}),
            }

        except (ValueError, AttributeError, RuntimeError) as exc:
            last_error = exc

    return {
        "latex_model": f"% generation failed: {last_error}",
        "latex_png_path": "",
        "costs": state.get("costs", {"total": 0.0, "by_agent": {}}),
    }
