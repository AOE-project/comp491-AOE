"""
ui/gradio/app.py — Gradio UI for AOE.

Professional dark-mode chat interface for MILP model refinement.
Runs everything in a single process.

Run with:
    python -m ui.gradio.app
"""
import csv
import io
import sys
from pathlib import Path
from typing import Optional

import gradio as gr
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))

from middleware.handle import AOEHandle

# ============ GLOBAL STATE ============
_aoe_handle   = AOEHandle()
_current_state: Optional[dict] = None

# ============ CORE HELPERS ============

def _build_bot_text(state: dict) -> str:
    # Chat mode: show the chat agent's response directly.
    if (state or {}).get("chat_mode") and (state or {}).get("chat_response"):
        return state["chat_response"]

    spec      = (state or {}).get("current_input_spec") or {}
    spec_type = spec.get("type", "")

    if spec_type == "set_size":
        error  = spec.get("error")
        prompt = spec.get("prompt", "")
        return f"**Error:** {error}\n\n{prompt}" if error else prompt

    if spec_type == "param_data":
        pname = spec.get("param_name", "parameter")
        error = spec.get("error")
        is_scalar = spec.get("is_scalar") or not spec.get("row_labels")
        if is_scalar:
            prompt = spec.get("prompt", f"Enter the value for **{pname}**. Type a single number.")
            return f"**Error collecting {pname}:** {error}\n\n{prompt}" if error else prompt
        if error:
            return (
                f"**Error collecting {pname}:** {error}\n\n"
                "Please resubmit using the data panel below."
            )
        return f"Please provide data for **{pname}** using the panel below."

    # Solver result available
    solver_result = (state.get("solver_result") or {})
    generated_code = (state.get("generated_code") or "").strip()
    if solver_result and generated_code:
        status  = solver_result.get("status", "")
        stdout  = (solver_result.get("stdout") or "").strip()
        stderr  = (solver_result.get("stderr") or "").strip()
        parts   = [f"**Generated Gurobi script:**\n\n```python\n{generated_code}\n```"]
        if status == "success":
            parts.append(f"**Result:**\n```\n{stdout}\n```")
        else:
            parts.append(f"**Run error ({status}):**\n```\n{stderr}\n```")
        return "\n\n".join(parts)

    # Code generation complete (solver not yet run)
    generated_code = (state.get("generated_code") or "").strip()
    if generated_code:
        syntax_error = state.get("code_syntax_error")
        if syntax_error:
            return f"**Code generated with syntax error:**\n> {syntax_error}\n\n```python\n{generated_code}\n```"
        return f"**Generated Gurobi script:**\n\n```python\n{generated_code}\n```"

    # Normal analyser mode
    parts   = []
    summary = (state.get("analysis_summary") or "").strip()
    if summary:
        parts.append(summary)
    questions = state.get("open_questions") or []
    if questions:
        parts.append("**Please clarify the following:**")
        for q in questions:
            parts.append(f"- {q}")
    return "\n\n".join(parts) if parts else "Processing..."


def _format_sidebar(state: dict):
    milp = state.get("milp_model") or {}
    if milp:
        obj        = milp.get("objective", {}).get("sense", "—")
        model_text = (
            f"| Metric | Value |\n"
            f"|--------|-------|\n"
            f"| Objective | `{obj}` |\n"
            f"| Sets | `{len(milp.get('sets', []))}` |\n"
            f"| Parameters | `{len(milp.get('parameters', []))}` |\n"
            f"| Variables | `{len(milp.get('variables', []))}` |\n"
        )
    else:
        model_text = "*Model not yet generated.*"

    def fmt(items, icon):
        if not items:
            return "*None yet.*"
        return "\n\n".join(f"{icon} {i}" for i in items)

    confirmed   = fmt(state.get("confirmed_assumptions",   []), "✔")
    unconfirmed = fmt(state.get("unconfirmed_assumptions", []), "?")
    return model_text, confirmed, unconfirmed


def _make_df_for_spec(spec: dict) -> tuple[Optional[pd.DataFrame], bool]:
    """
    Build an empty DataFrame matching the spec's shape.
    Returns (df, show) — show=False when max dimension > 10.
    """
    row_labels = spec.get("row_labels") or []
    col_labels = spec.get("col_labels") or None
    param_name = spec.get("param_name", "value")

    if not row_labels:
        return None, False

    if col_labels:
        if max(len(row_labels), len(col_labels)) > 10:
            return None, False
        df = pd.DataFrame(
            [[None] * len(col_labels) for _ in row_labels],
            index=pd.Index(row_labels, name=""),
            columns=col_labels,
        )
    else:
        if len(row_labels) > 10:
            return None, False
        df = pd.DataFrame(
            [[None] for _ in row_labels],
            index=pd.Index(row_labels, name=""),
            columns=[param_name],
        )
    # reset_index moves row labels from the (hidden) pandas index into a
    # visible first column so Gradio renders them in the table.
    return df.reset_index(), True


def _df_to_csv_str(df_value) -> str:
    """Serialize a Gradio Dataframe value to a CSV string (header only, no index)."""
    if isinstance(df_value, pd.DataFrame):
        return df_value.to_csv(index=False, header=True)
    if isinstance(df_value, list):
        buf = io.StringIO()
        writer = csv.writer(buf)
        for row in df_value:
            writer.writerow(row)
        return buf.getvalue()
    return str(df_value)


# ============ SHARED OUTPUT BUILDER ============
# All callbacks share the same 12-element output tuple:
#   msg_input, chatbot, approve_row, msg_row,
#   data_panel, data_prompt_md, data_error_md,
#   param_df, param_file,
#   model_display, confirmed_display, unconfirmed_display

def _build_all_outputs(state: dict, chat_history: list, clear_msg: bool = False):
    spec      = (state or {}).get("current_input_spec") or {}
    spec_type = spec.get("type", "")
    is_param  = spec_type == "param_data"
    is_scalar = is_param and (spec.get("is_scalar") or not spec.get("row_labels"))
    is_tabular = is_param and not is_scalar

    bot_text    = _build_bot_text(state)
    # Preserve the full conversation — append bot reply to whatever is already displayed.
    new_history = list(chat_history) + [{"role": "assistant", "content": bot_text}]

    questions        = (state or {}).get("open_questions") or []
    not_approved_yet = not (state or {}).get("analyser_approved", False)
    has_run          = (state or {}).get("iteration_count", 0) > 0
    approve_vis = gr.update(visible=has_run and not questions and not spec_type and not_approved_yet)

    # Scalars are collected via the normal chat input, not the data panel
    msg_row_vis   = gr.update(visible=not is_tabular)
    data_panel_vis = gr.update(visible=is_tabular)

    prompt = spec.get("prompt", "") if is_tabular else ""

    error      = spec.get("error") or "" if is_tabular else ""
    error_upd  = gr.update(
        visible=bool(error),
        value=f"> **Error:** {error}" if error else "",
    )

    if is_tabular:
        df, show_df = _make_df_for_spec(spec)
        df_upd = gr.update(visible=show_df, value=df) if show_df else gr.update(visible=False, value=None)
    else:
        # Clear any stale table from a previous param
        df_upd = gr.update(visible=False, value=None)

    file_upd = gr.update(value=None)

    model_info, conf, unconf = _format_sidebar(state)

    return (
        "" if clear_msg else gr.update(),   # msg_input
        new_history,                         # chatbot
        approve_vis,                         # approve_row
        msg_row_vis,                         # msg_row
        data_panel_vis,                      # data_panel
        prompt,                              # data_prompt_md
        error_upd,                           # data_error_md
        df_upd,                              # param_df
        file_upd,                            # param_file
        model_info,                          # model_display
        conf,                                # confirmed_display
        unconf,                              # unconfirmed_display
    )


# ============ CALLBACKS ============

def _process_message(user_message: str, chat_history: list):
    """Handle a normal chat / set-size answer turn."""
    global _current_state
    if not user_message.strip():
        return (gr.update(),) * 12

    # Add the user's message to the display before the bot responds.
    chat_history = list(chat_history) + [{"role": "user", "content": user_message}]

    try:
        _current_state = _aoe_handle.run(user_message, _current_state)
        return _build_all_outputs(_current_state, chat_history, clear_msg=True)
    except Exception as e:
        chat_history.append({"role": "assistant", "content": f"**Error:** {str(e)}"})
        return ("", chat_history) + (gr.update(),) * 10


def _approve(chat_history: list):
    """User clicked Approve — set analyser_approved and advance graph."""
    global _current_state
    if _current_state is None:
        return (gr.update(),) * 12

    _current_state["analyser_approved"] = True
    _current_state = _aoe_handle.run("__approved__", _current_state)
    return _build_all_outputs(_current_state, chat_history)


def _request_changes(chat_history: list):
    """User clicked Request Changes — hide approve row, prompt for feedback."""
    chat_history.append({"role": "assistant", "content": "Sure — what would you like to change?"})
    return (
        gr.update(),                # msg_input
        chat_history,               # chatbot
        gr.update(visible=False),   # approve_row
        gr.update(visible=True),    # msg_row
        gr.update(visible=False),   # data_panel
        gr.update(),                # data_prompt_md
        gr.update(),                # data_error_md
        gr.update(),                # param_df
        gr.update(),                # param_file
        gr.update(),                # model_display
        gr.update(),                # confirmed_display
        gr.update(),                # unconfirmed_display
    )


def _submit_param_data(file, df_value, chat_history: list):
    """User submitted parameter data via file upload or the editable table."""
    global _current_state
    spec = (_current_state or {}).get("current_input_spec") or {}

    if file is not None:
        answer = file if isinstance(file, str) else file.name
    elif df_value is not None:
        answer = _df_to_csv_str(df_value)
    else:
        err_spec = {**spec, "error": "Please upload a CSV file or fill in the table above."}
        _current_state = {**(_current_state or {}), "current_input_spec": err_spec}
        return _build_all_outputs(_current_state, chat_history)

    try:
        _current_state = _aoe_handle.run(answer, _current_state)
        return _build_all_outputs(_current_state, chat_history)
    except Exception as e:
        chat_history.append({"role": "assistant", "content": f"**Error:** {str(e)}"})
        return (gr.update(),) * 12


def _reset_session():
    global _current_state
    _current_state = None
    return (
        "",                          # msg_input
        [],                          # chatbot
        gr.update(visible=False),    # approve_row
        gr.update(visible=True),     # msg_row
        gr.update(visible=False),    # data_panel
        "",                          # data_prompt_md
        gr.update(visible=False, value=""),  # data_error_md
        gr.update(visible=False),    # param_df
        gr.update(value=None),       # param_file
        "*Waiting for input...*",    # model_display
        "*None yet.*",               # confirmed_display
        "*None yet.*",               # unconfirmed_display
    )


# ============ CSS ============

custom_css = (Path(__file__).parent / "style.css").read_text(encoding="utf-8")

# ============ GRADIO UI ============

with gr.Blocks(title="AOE — Automated Optimization Engineer", css=custom_css) as demo:

    gr.Markdown("<p class='aoe-title'>AOE &nbsp;/&nbsp; Automated Optimization Engineer</p>")

    with gr.Row(equal_height=True):

        # LEFT: Chat + input area
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="", height=460, show_label=False)

            # Approve / Request Changes row — hidden until open_questions is empty
            with gr.Row(visible=False, elem_classes=["approve-row"]) as approve_row:
                gr.Markdown("**Does this model look correct?**", elem_classes=["approve-label"])
                approve_btn = gr.Button("✔ Approve",         elem_classes=["approve-btn"], scale=1)
                changes_btn = gr.Button("✎ Request Changes", elem_classes=["changes-btn"], scale=1)

            # Data collection panel — shown during param_data collection turns
            with gr.Group(visible=False, elem_classes=["data-panel"]) as data_panel:
                data_prompt_md = gr.Markdown("", elem_classes=["data-prompt"])
                data_error_md  = gr.Markdown("", visible=False, elem_classes=["data-error"])
                with gr.Row():
                    param_file = gr.File(
                        label="Upload CSV",
                        file_types=[".csv"],
                        scale=1,
                    )
                    param_df = gr.Dataframe(
                        label="Or edit manually (available when each dimension ≤ 10)",
                        interactive=True,
                        visible=False,
                        scale=2,
                    )
                data_submit_btn = gr.Button(
                    "Submit Data",
                    variant="primary",
                    elem_classes=["data-submit-btn"],
                )

            # Normal chat input row — hidden during param_data collection
            with gr.Row() as msg_row:
                msg_input = gr.Textbox(
                    show_label=False,
                    placeholder="Describe your optimization problem...",
                    scale=9, lines=1, max_lines=1,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1, min_width=80)

        # RIGHT: Sidebar
        with gr.Column(scale=1, min_width=230):
            gr.Markdown("### Model Structure")
            model_display = gr.Markdown(value="*Waiting for input...*", elem_classes=["sidebar-card"])

            gr.Markdown("### Confirmed")
            confirmed_display = gr.Markdown(value="*None yet.*", elem_classes=["sidebar-card"])

            gr.Markdown("### Pending Review")
            unconfirmed_display = gr.Markdown(value="*None yet.*", elem_classes=["sidebar-card"])

            reset_btn = gr.Button("New Session", variant="secondary")

    # ── All callbacks share the same 12-element output list ──────────────
    ALL_OUTPUTS = [
        msg_input,
        chatbot,
        approve_row,
        msg_row,
        data_panel,
        data_prompt_md,
        data_error_md,
        param_df,
        param_file,
        model_display,
        confirmed_display,
        unconfirmed_display,
    ]

    send_btn.click(
        fn=_process_message,
        inputs=[msg_input, chatbot],
        outputs=ALL_OUTPUTS,
    )
    msg_input.submit(
        fn=_process_message,
        inputs=[msg_input, chatbot],
        outputs=ALL_OUTPUTS,
    )

    approve_btn.click(
        fn=_approve,
        inputs=[chatbot],
        outputs=ALL_OUTPUTS,
    )
    changes_btn.click(
        fn=_request_changes,
        inputs=[chatbot],
        outputs=ALL_OUTPUTS,
    )

    data_submit_btn.click(
        fn=_submit_param_data,
        inputs=[param_file, param_df, chatbot],
        outputs=ALL_OUTPUTS,
    )

    reset_btn.click(
        fn=_reset_session,
        outputs=ALL_OUTPUTS,
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861)
