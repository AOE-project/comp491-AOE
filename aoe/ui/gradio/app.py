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
_attached_file_path: Optional[str] = None

# ============ CORE HELPERS ============

def _build_bot_text(state: dict) -> str:
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

    # CRITICAL: If analyser is NOT approved yet, show questions/summary only
    # Don't show code until after approval and input retrieval
    is_approved = (state or {}).get("analyser_approved", False)
    if not is_approved:
        # Still in Analyser phase - show questions and summary
        parts = []
        summary = ((state or {}).get("analysis_summary") or "").strip()
        if summary:
            parts.append(summary)
        questions = (state or {}).get("open_questions", [])
        if questions:
            parts.append("**Please clarify the following:**")
            for q in questions:
                parts.append(f"- {q}")
        return "\n\n".join(parts) if parts else "Processing..."

    # Analyser approved - now show code/solver results if available
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
            # Get error type for display
            error_type = state.get("last_error_type", "unknown_error")
            error_label = error_type.replace("_", " ").title()
            
            # Format as Markdown for Gradio Chatbot compatibility
            error_md = f"> **Solver Error ({error_label})**\n> \n> {stderr}"
            parts.append(error_md)
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


def _build_pipeline_status(state: dict) -> str:
    """
    Determine which agents have completed and which is currently active.
    Returns an HTML visualization of the pipeline.
    
    CRITICAL LOGIC:
    - Analyser completes ONLY when: analysis exists AND approver approved AND no questions pending
    - Otherwise, Analyser is ACTIVE (awaiting user approval or answers)
    """
    if not state:
        state = {}
    
    agents = [
        ("Analyser", "analysis_summary"),
        ("Input Retrieval", "input_retrieval_cursor"),
        ("Code Generator", "generated_code"),
        ("Solver", "solver_result"),
        ("Debug", "last_error_type"),
        ("Explainer", "explanation"),
    ]
    
    # Determine which agents are completed and which is active
    completed = set()
    active = -1
    
    # Check analyser approval state
    analysis_summary = (state.get("analysis_summary") or "").strip()
    is_approved = state.get("analyser_approved", False)
    open_questions = state.get("open_questions", [])
    has_error_before_code = state.get("last_execution_error") and not state.get("generated_code")
    
    # ANALYSER phase - only complete if APPROVED and NO QUESTIONS PENDING
    if analysis_summary and is_approved and not open_questions:
        completed.add(0)  # Analyser truly done
        
        # INPUT RETRIEVAL phase
        if not state.get("current_input_spec"):
            # No pending data → input retrieval is done
            completed.add(1)
            
            # CODE GENERATOR phase
            if state.get("generated_code"):
                completed.add(2)
                
                # SOLVER phase
                if state.get("solver_result"):
                    completed.add(3)
                    
                    # ERROR or SUCCESS path
                    if state.get("last_execution_error"):
                        # DEBUG phase
                        if state.get("last_error_type"):
                            completed.add(4)
                            
                            # Still has error after debug
                            if state.get("explanation"):
                                completed.add(5)
                            else:
                                active = 5  # Explainer (final)
                        else:
                            active = 4  # Debug classification happening
                    else:
                        # Success path - go to explainer
                        if state.get("explanation"):
                            completed.add(5)
                        else:
                            active = 5  # Explainer running
                else:
                    active = 3  # Solver running
            else:
                active = 2  # Code Generator running
        else:
            # Pending data needed → Input Retrieval waiting
            active = 1
    elif has_error_before_code:
        # Error occurred during Analyser phase
        active = 0
    elif analysis_summary:
        # Analysis exists but NOT YET APPROVED or still has questions → Analyser is ACTIVE
        active = 0
    else:
        # No analysis yet - initial state, no agent active
        active = -1
    
    # Build HTML pipeline
    html = '<div style="display: flex; gap: 8px; margin: 12px 0; flex-wrap: wrap; align-items: center;">'
    
    for i, (name, _) in enumerate(agents):
        if i in completed:
            status = "completed"
            bg_color = "#10b981"  # Green
            text_color = "white"
            icon = "✓"
        elif i == active:
            status = "active"
            bg_color = "#f59e0b"  # Amber
            text_color = "white"
            icon = "●"
        else:
            status = "pending"
            bg_color = "#6b7280"  # Gray
            text_color = "#9ca3af"
            icon = "○"
        
        html += f'''
        <div style="
            background-color: {bg_color};
            color: {text_color};
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 500;
            min-width: 100px;
            text-align: center;
        ">
            <span style="margin-right: 4px;">{icon}</span>{name}
        </div>
        '''
        
        # Add arrow between agents
        if i < len(agents) - 1:
            html += f'''<div style="color: #9ca3af; font-size: 16px;">→</div>'''
    
    html += '</div>'
    return html


# ============ SHARED OUTPUT BUILDER ============
# All callbacks share the same 12-element output tuple:
#   msg_input, chatbot, approve_row, msg_row,
#   data_panel, data_prompt_md, error_md, param_file,
#   model_display, confirmed_display, unconfirmed_display, pipeline_status

def _build_all_outputs(state: dict, chat_history: list, clear_msg: bool = False):
    spec      = (state or {}).get("current_input_spec") or {}
    spec_type = spec.get("type", "")
    is_param  = spec_type == "param_data"
    is_scalar = is_param and (spec.get("is_scalar") or not spec.get("row_labels"))

    bot_text    = _build_bot_text(state)
    # Preserve the full conversation — append bot reply to whatever is already displayed.
    new_history = list(chat_history) + [{"role": "assistant", "content": bot_text}]

    questions        = (state or {}).get("open_questions") or []
    not_approved_yet = not (state or {}).get("analyser_approved", False)
    has_run          = (state or {}).get("iteration_count", 0) > 0
    approve_vis = gr.update(visible=has_run and not questions and not spec_type and not_approved_yet)

    # Scalars are collected via the normal chat input, not the data panel
    msg_row_vis   = gr.update(visible=True)  # Always visible for attachment button access
    # Data panel is hidden — file uploads handled via attach button
    data_panel_vis = gr.update(visible=False)

    prompt = spec.get("prompt", "") if is_param else ""

    error      = spec.get("error") or "" if is_param else ""
    error_upd  = gr.update(
        visible=bool(error),
        value=f"> **Error:** {error}" if error else "",
    )

    model_info, conf, unconf = _format_sidebar(state)
    pipeline_html = _build_pipeline_status(state)

    # Disable message input during param_data collection (user should use attachment button)
    msg_input_upd = gr.update(value="", interactive=False) if (clear_msg and is_param) else (
        gr.update(value="") if clear_msg else gr.update(interactive=not is_param)
    )

    return (
        msg_input_upd,               # msg_input
        new_history,                         # chatbot
        approve_vis,                         # approve_row
        msg_row_vis,                         # msg_row
        data_panel_vis,                      # data_panel
        prompt,                              # data_prompt_md
        error_upd,                           # data_error_md
        gr.update(value=None),               # param_file
        model_info,                          # model_display
        conf,                                # confirmed_display
        unconf,                              # unconfirmed_display
        pipeline_html,                       # pipeline_status
    )


# ============ CALLBACKS ============

def _handle_attach(file):
    """Handle file upload and update placeholder."""
    global _attached_file_path
    _attached_file_path = file.name if file else None
    name = Path(file.name).name if file else ""
    placeholder = f"📎 {name}  —  Describe your problem..." if name else "Describe your optimization problem..."
    return gr.update(placeholder=placeholder)

def _process_message(user_message: str, chat_history: list):
    """Handle a normal chat / set-size answer turn, including file uploads."""
    global _current_state, _attached_file_path
    
    if not user_message.strip() and _attached_file_path is None:
        return (gr.update(),) * 12

    # Get filename for display
    display_content = user_message
    if not user_message.strip() and _attached_file_path:
        # Show the filename instead of "(attached file)"
        file_name = Path(_attached_file_path).name
        display_content = f"📎 {file_name}"
    
    # Add the user's message to the display before the bot responds.
    chat_history = list(chat_history) + [{"role": "user", "content": display_content}]

    spec = (_current_state or {}).get("current_input_spec") or {}
    spec_type = spec.get("type", "")

    # During param_data collection, use attached file if available
    if spec_type == "param_data" and _attached_file_path:
        answer = _attached_file_path
        _attached_file_path = None
    else:
        answer = user_message

    try:
        _current_state = _aoe_handle.run(answer, _current_state)
        _attached_file_path = None  # Clear after use
        return _build_all_outputs(_current_state, chat_history, clear_msg=True)
    except Exception as e:
        chat_history.append({"role": "assistant", "content": f"**Error:** {str(e)}"})
        _attached_file_path = None
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
        gr.update(interactive=True),  # msg_input - enable for user input
        chat_history,               # chatbot
        gr.update(visible=False),   # approve_row
        gr.update(visible=True),    # msg_row
        gr.update(visible=False),   # data_panel
        gr.update(),                # data_prompt_md
        gr.update(),                # data_error_md
        gr.update(value=None),      # param_file
        gr.update(),                # model_display
        gr.update(),                # confirmed_display
        gr.update(),                # unconfirmed_display
        gr.update(),                # pipeline_status
    )


def _submit_param_data(file, chat_history: list):
    """User submitted parameter data via CSV file upload."""
    global _current_state
    spec = (_current_state or {}).get("current_input_spec") or {}

    if file is not None:
        answer = file if isinstance(file, str) else file.name
    else:
        err_spec = {**spec, "error": "Please upload a CSV file."}
        _current_state = {**(_current_state or {}), "current_input_spec": err_spec}
        return _build_all_outputs(_current_state, chat_history)

    try:
        _current_state = _aoe_handle.run(answer, _current_state)
        return _build_all_outputs(_current_state, chat_history)
    except Exception as e:
        chat_history.append({"role": "assistant", "content": f"**Error:** {str(e)}"})
        return (gr.update(),) * 12


def _reset_session():
    global _current_state, _attached_file_path
    _current_state = None
    _attached_file_path = None
    return (
        gr.update(value="", interactive=True),  # msg_input - reset and enable
        [{
            "role": "assistant",
            "content": "👋 **Welcome to AOE** — Automated Optimization Engineer\n\nDescribe your optimization problem in natural language. I'll help you build and solve it!"
        }],                          # chatbot (reset to initial message)
        gr.update(visible=False),    # approve_row
        gr.update(visible=True),     # msg_row
        gr.update(visible=False),    # data_panel
        "",                          # data_prompt_md
        gr.update(visible=False, value=""),  # data_error_md
        gr.update(value=None),       # param_file
        "*Awaiting problem description...*",    # model_display
        "*No assumptions confirmed yet.*",      # confirmed_display
        "*No assumptions to review yet.*",      # unconfirmed_display
        _build_pipeline_status({}),  # pipeline_status - consistent with initial state
    )


# ============ CSS ============

custom_css = (Path(__file__).parent / "style.css").read_text(encoding="utf-8")

# ============ GRADIO UI ============

with gr.Blocks(title="AOE — Automated Optimization Engineer", css=custom_css) as demo:

    gr.Markdown("<p class='aoe-title'>AOE &nbsp;/&nbsp; Automated Optimization Engineer</p>")
    
    # Pipeline status indicator
    pipeline_status = gr.HTML(
        value='<div style="text-align: center; color: #9ca3af; font-size: 12px;">Pipeline ready</div>',
        elem_classes=["pipeline-status"]
    )

    with gr.Row(equal_height=True):

        # LEFT: Chat + input area
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="", 
                height=460, 
                show_label=False,
                value=[{
                    "role": "assistant",
                    "content": "👋 **Welcome to AOE** — Automated Optimization Engineer\n\nDescribe your optimization problem in natural language. I'll help you build and solve it!"
                }]
            )

            # Approve / Request Changes row — hidden until open_questions is empty
            with gr.Row(visible=False, elem_classes=["approve-row"]) as approve_row:
                gr.Markdown("**Does this model look correct?**", elem_classes=["approve-label"])
                approve_btn = gr.Button("✔ Approve",         elem_classes=["approve-btn"], scale=1)
                changes_btn = gr.Button("✎ Request Changes", elem_classes=["changes-btn"], scale=1)

            # Data collection panel — shown during param_data collection turns
            with gr.Group(visible=False, elem_classes=["data-panel"]) as data_panel:
                data_prompt_md = gr.Markdown("", elem_classes=["data-prompt"])
                data_error_md  = gr.Markdown("", visible=False, elem_classes=["data-error"])
                param_file = gr.File(
                    label="Upload CSV",
                    file_types=[".csv"],
                )
                data_submit_btn = gr.Button(
                    "Submit Data",
                    variant="primary",
                    elem_classes=["data-submit-btn"],
                )

            # Normal chat input row — hidden during param_data collection
            with gr.Row() as msg_row:
                attach_btn = gr.UploadButton(
                    "📎",
                    file_types=[".csv"],
                    scale=1,
                    min_width=48,
                    variant="secondary",
                    elem_classes=["attach-btn"],
                )
                msg_input = gr.Textbox(
                    show_label=False,
                    placeholder="Describe your optimization problem...",
                    scale=9, lines=1, max_lines=1,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1, min_width=80)

        # RIGHT: Sidebar
        with gr.Column(scale=1, min_width=230):
            gr.Markdown("### Model Structure")
            model_display = gr.Markdown(
                value="*Awaiting problem description...*",
                elem_classes=["sidebar-card"]
            )

            gr.Markdown("### Confirmed")
            confirmed_display = gr.Markdown(
                value="*No assumptions confirmed yet.*",
                elem_classes=["sidebar-card"]
            )

            gr.Markdown("### Pending Review")
            unconfirmed_display = gr.Markdown(
                value="*No assumptions to review yet.*",
                elem_classes=["sidebar-card"]
            )

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
        param_file,
        model_display,
        confirmed_display,
        unconfirmed_display,
        pipeline_status,
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

    attach_btn.upload(
        fn=_handle_attach,
        inputs=[attach_btn],
        outputs=[msg_input],
    )

    data_submit_btn.click(
        fn=_submit_param_data,
        inputs=[param_file, chatbot],
        outputs=ALL_OUTPUTS,
    )

    reset_btn.click(
        fn=_reset_session,
        outputs=ALL_OUTPUTS,
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861)
