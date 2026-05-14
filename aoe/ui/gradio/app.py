"""
ui/gradio/app.py — Gradio UI for AOE.

Professional dark-mode chat interface for MILP model refinement.
Runs everything in a single process.

Run with:
    python -m ui.gradio.app
"""
import csv
import html
import io
import sys
from pathlib import Path
from typing import Optional

import gradio as gr
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))

from middleware.handle import AOEHandle
from core.logger import SessionLogger

# ============ GLOBAL STATE ============
_aoe_handle   = AOEHandle()
_current_state: Optional[dict] = None

# ============ CORE HELPERS ============

_STATUS_PRESENTATION = {
    "optimal":    ("result-success",   "&#10003;", "Optimal"),
    "infeasible": ("result-error",     "&#10007;", "Infeasible"),
    "unbounded":  ("result-unbounded", "&#8734;",  "Unbounded"),
}


def _format_value(v) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (int,)):
        return str(v)
    if isinstance(v, float):
        if abs(v - round(v)) < 1e-9 and abs(v) < 1e15:
            return f"{int(round(v))}"
        return f"{v:.4f}".rstrip("0").rstrip(".")
    return html.escape(str(v))


def _render_solver_result(solver_result: dict) -> str:
    """Render solver_result as a status badge + objective + variables table."""
    status = (solver_result.get("status") or "other:unknown").lower()
    obj_value = solver_result.get("objective_value")
    variables = solver_result.get("variables") or {}
    iis_path = solver_result.get("iis_path")
    stderr = (solver_result.get("stderr") or "").strip()

    # Pick presentation by status family
    if status in _STATUS_PRESENTATION:
        block_class, icon, label = _STATUS_PRESENTATION[status]
    elif status.startswith("other"):
        block_class, icon, label = "result-other", "&#9888;", status.replace("other:", "Status ").title()
    else:
        block_class, icon, label = "result-error", "&#10007;", "Error"

    parts = [
        f"<div class='result-block {block_class}'>",
        f"<span class='result-badge'>{icon} {html.escape(label)}</span>",
    ]

    if status == "optimal":
        if obj_value is not None:
            parts.append(
                "<div class='result-objective'>"
                "<span class='result-objective-label'>Objective</span>"
                f"<span class='result-objective-value'>{_format_value(obj_value)}</span>"
                "</div>"
            )
        if variables:
            rows = "".join(
                f"<tr><td class='var-name'>{html.escape(str(name))}</td>"
                f"<td class='var-value'>{_format_value(value)}</td></tr>"
                for name, value in variables.items()
            )
            parts.append(
                "<table class='result-table'>"
                "<thead><tr><th>Variable</th><th>Value</th></tr></thead>"
                f"<tbody>{rows}</tbody>"
                "</table>"
            )
        else:
            parts.append(
                "<div class='result-note'>No non-zero decision variables.</div>"
            )

    elif status == "infeasible":
        parts.append(
            "<div class='result-note'>"
            "The model has no feasible solution. "
            "An irreducible inconsistent subsystem (IIS) was written to "
            f"<code>{html.escape(iis_path) if iis_path else 'iis.ilp'}</code>."
            "</div>"
        )

    elif status == "unbounded":
        parts.append(
            "<div class='result-note'>"
            "The objective is unbounded — no finite optimum exists for this formulation."
            "</div>"
        )

    else:
        # runtime_error / timeout / other
        msg = stderr.splitlines()[-1] if stderr else "Solver did not produce a result."
        parts.append(
            f"<div class='result-note'><pre class='result-pre'>{html.escape(msg)}</pre></div>"
        )

    parts.append("</div>")
    return "".join(parts)


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
        code_accordion = (
            "<details class='code-accordion'>\n"
            "<summary><b>Generated Gurobi Script</b></summary>\n\n"
            f"```python\n{generated_code}\n```\n"
            "</details>"
        )
        return f"{code_accordion}\n\n{_render_solver_result(solver_result)}"

    # Code generation complete (solver not yet run)
    generated_code = (state.get("generated_code") or "").strip()
    if generated_code:
        syntax_error = state.get("code_syntax_error")
        if syntax_error:
            return (
                f"**Code generated with syntax error:**\n> {syntax_error}\n\n"
                "<details class='code-accordion'>\n"
                "<summary><b>Generated Gurobi Script</b></summary>\n\n"
                f"```python\n{generated_code}\n```\n"
                "</details>"
            )
        return (
            "<details class='code-accordion'>\n"
            "<summary><b>Generated Gurobi Script</b></summary>\n\n"
            f"```python\n{generated_code}\n```\n"
            "</details>"
        )

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
    png_path = state.get("latex_png_path") or None

    def fmt(items, icon):
        if not items:
            return "*None yet.*"
        return "\n\n".join(f"{icon} {i}" for i in items)

    confirmed   = fmt(state.get("confirmed_assumptions",   []), "✔")
    unconfirmed = fmt(state.get("unconfirmed_assumptions", []), "?")
    return png_path, confirmed, unconfirmed


def _make_df_for_spec(
    spec: dict,
    existing_data=None,
) -> tuple[Optional[pd.DataFrame], bool]:
    """
    Build a DataFrame matching the spec's shape.
    Pre-fills cells from existing_data (raw_data value for this parameter)
    so users only need to fill in new or changed rows.
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
        rows = []
        for r in row_labels:
            row = []
            for c in col_labels:
                val = None
                if isinstance(existing_data, dict):
                    r_data = existing_data.get(r)
                    if isinstance(r_data, dict):
                        val = r_data.get(c)
                row.append(val)
            rows.append(row)
        df = pd.DataFrame(
            rows,
            index=pd.Index(row_labels, name=""),
            columns=col_labels,
        )
    else:
        if len(row_labels) > 10:
            return None, False
        rows = []
        for r in row_labels:
            val = None
            if isinstance(existing_data, dict):
                val = existing_data.get(r)
            rows.append([val])
        df = pd.DataFrame(
            rows,
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


def _history_to_chatbot(history: list) -> list:
    """Convert state history to Gradio chatbot format, filtering internal signals."""
    return [
        {"role": msg["role"], "content": msg["content"]}
        for msg in (history or [])
        if msg.get("role") in ("user", "assistant")
        and not (msg.get("role") == "user" and msg.get("content") == "__approved__")
    ]


# ============ SHARED OUTPUT BUILDER ============
# All callbacks share the same 12-element output tuple:
#   msg_input, chatbot, approve_row, msg_row,
#   data_panel, data_prompt_md, data_error_md,
#   param_df, param_file,
#   model_display, confirmed_display, unconfirmed_display

def _build_all_outputs(
    state: dict,
    chat_history: list,
    clear_msg: bool = False,
    skip_bot_message: bool = False,
):
    spec      = (state or {}).get("current_input_spec") or {}
    spec_type = spec.get("type", "")
    is_param  = spec_type == "param_data"
    is_scalar = is_param and (spec.get("is_scalar") or not spec.get("row_labels"))
    is_tabular = is_param and not is_scalar

    bot_text    = _build_bot_text(state)
    # On resume, chat_history already contains the last assistant turn — don't
    # re-append a fresh bot_text or we'd duplicate it.
    # Tabular data-collection turns show their prompt in the data panel — don't
    # pollute the transcript with "Please provide data for X" messages.
    if skip_bot_message or is_tabular:
        new_history = list(chat_history)
    else:
        new_history = list(chat_history) + [{"role": "assistant", "content": bot_text}]

    questions        = (state or {}).get("open_questions") or []
    not_approved_yet = not (state or {}).get("analyser_approved", False)
    has_run          = (state or {}).get("iteration_count", 0) > 0
    _show_approve = has_run and not questions and not spec_type and not_approved_yet
    approve_vis = gr.update(visible=_show_approve)

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
        data_key     = spec.get("data_key")
        existing_val = ((state or {}).get("raw_data") or {}).get(data_key)
        df, show_df  = _make_df_for_spec(spec, existing_data=existing_val)
        df_upd   = gr.update(visible=show_df, value=df) if show_df else gr.update(visible=False, value=None)
        file_upd = gr.update(visible=True, value=None)   # always offer upload during collection
    else:
        # Clear any stale table from a previous param; hide upload outside data-collection
        df_upd   = gr.update(visible=False, value=None)
        file_upd = gr.update(visible=False, value=None)

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


# ============ SESSION HELPERS ============

def _load_session_list():
    """Return updated Radio choices from saved sessions, newest first.

    Preserve the currently-active session_id as the selected value so the
    sidebar row stays highlighted across re-renders.
    """
    sessions = SessionLogger.list_sessions()
    choices = []
    for s in sessions:
        raw_title = s.get("title") or s.get("problem_description", "Untitled")
        title = raw_title[:38] + "…" if len(raw_title) > 38 else raw_title
        date_str = (s.get("last_updated") or "")[:16].replace("T", " ")
        label = f"{title}\n{date_str}" if date_str else title
        choices.append((label, s["session_id"]))
    active = (_current_state or {}).get("session_id")
    valid_ids = {sid for _, sid in choices}
    selected = active if active in valid_ids else None
    return gr.update(choices=choices, value=selected)


# ============ CALLBACKS ============

def _process_message(user_message: str, chat_history: list):
    global _current_state, _aoe_handle

    user_message = (user_message or "").strip()
    if not user_message:
        return (gr.update(),) * 12

    chat_history = list(chat_history or []) + [
        {"role": "user", "content": user_message}
    ]

    try:
        _current_state = _aoe_handle.run(user_message, _current_state)
        outputs = _build_all_outputs(_current_state, chat_history, clear_msg=True)
        _current_state["chat_history"] = outputs[1]
        _aoe_handle.save(_current_state)
        return outputs
    except Exception as e:
        chat_history.append({"role": "assistant", "content": f"**Error:** {str(e)}"})
        return ("", chat_history) + (gr.update(),) * 10

def _refresh_approve_row():
    state = _current_state or {}
    spec = state.get("current_input_spec") or {}
    questions = state.get("open_questions") or []

    show = (
        state.get("iteration_count", 0) > 0
        and not questions
        and not spec.get("type")
        and not state.get("analyser_approved", False)
    )

    return gr.update(visible=show)

def _approve(chat_history: list):
    """User clicked Approve — set analyser_approved and advance graph."""
    global _current_state
    if _current_state is None:
        return (gr.update(),) * 12

    _current_state["analyser_approved"] = True
    _current_state = _aoe_handle.run("__approved__", _current_state)
    outputs = _build_all_outputs(_current_state, chat_history)
    _current_state["chat_history"] = outputs[1]
    _aoe_handle.save(_current_state)
    return outputs


def _request_changes(chat_history: list):
    """User clicked Request Changes — clear latex, hide approve row, prompt for feedback."""
    global _current_state
    if _current_state:
        _current_state["latex_model"]    = ""
        _current_state["latex_png_path"] = ""
    chat_history.append({"role": "assistant", "content": "Sure — what would you like to change?"})
    if _current_state:
        _current_state["chat_history"] = chat_history
        _aoe_handle.save(_current_state)
    return (
        gr.update(),                # msg_input
        chat_history,               # chatbot
        gr.update(visible=False),   # approve_row
        gr.update(visible=True),    # msg_row
        gr.update(visible=False),   # data_panel
        gr.update(),                # data_prompt_md
        gr.update(),                # data_error_md
        gr.update(),                # param_df
        gr.update(visible=False),   # param_file
        gr.update(value=None),      # model_display — clear stale PNG
        gr.update(),                # confirmed_display
        gr.update(),                # unconfirmed_display
    )


def _submit_param_data(file, df_value, chat_history: list):
    """User submitted parameter data via file upload or the editable table."""
    global _current_state
    spec = (_current_state or {}).get("current_input_spec") or {}

    if file is not None:
        if isinstance(file, str):
            answer = file
        elif isinstance(file, dict):
            answer = file.get("name", str(file))
        elif hasattr(file, "name"):
            answer = file.name
        else:
            answer = str(file)
    elif df_value is not None:
        answer = _df_to_csv_str(df_value)
    else:
        err_spec = {**spec, "error": "Please upload a CSV file or fill in the table above."}
        _current_state = {**(_current_state or {}), "current_input_spec": err_spec}
        return _build_all_outputs(_current_state, chat_history)

    try:
        _current_state = _aoe_handle.run(answer, _current_state)
        outputs = _build_all_outputs(_current_state, chat_history)
        _current_state["chat_history"] = outputs[1]
        _aoe_handle.save(_current_state)
        return outputs
    except Exception as e:
        chat_history.append({"role": "assistant", "content": f"**Error:** {str(e)}"})
        return (gr.update(),) * 12


def _reset_session():
    global _current_state, _aoe_handle
    _current_state = None
    _aoe_handle = AOEHandle()
    return (
        "",                          # msg_input
        [],                          # chatbot
        gr.update(visible=False),    # approve_row
        gr.update(visible=True),     # msg_row
        gr.update(visible=False),    # data_panel
        "",                          # data_prompt_md
        gr.update(visible=False, value=""),  # data_error_md
        gr.update(visible=False),                # param_df
        gr.update(visible=False, value=None),    # param_file
        None,                                # model_display
        "*None yet.*",               # confirmed_display
        "*None yet.*",               # unconfirmed_display
    )


def _resume_session(session_id: str):
    """Load a saved session and rebuild the UI."""
    global _aoe_handle, _current_state
    if not session_id:
        return (gr.update(),) * 12
    try:
        _aoe_handle, _current_state = AOEHandle.resume(session_id)
    except Exception:
        return (gr.update(),) * 12
    chat_history = _current_state.get("chat_history") or []
    return _build_all_outputs(_current_state, chat_history, skip_bot_message=True)


def _toggle_sessions(is_open: bool):
    new_open = not is_open
    return gr.update(visible=new_open), new_open


# ============ CSS ============

custom_css = (Path(__file__).parent / "style.css").read_text(encoding="utf-8")

# ============ GRADIO UI ============

with gr.Blocks(title="AOE — Automated Optimization Engineer") as demo:

    sessions_open_state = gr.State(value=False)

    # Header row: sessions toggle + title + spacer
    with gr.Row(elem_id="aoe-header"):
        toggle_btn = gr.Button("☰", elem_classes=["sessions-toggle-btn"], scale=0, min_width=44)
        gr.Markdown("<p class='aoe-title'>AOE &nbsp;/&nbsp; Automated Optimization Engineer</p>")
        gr.HTML("<div style='min-width:44px'></div>")   # balances the toggle button

    with gr.Row(equal_height=True):

        # LEFT: Sessions sidebar
        with gr.Column(scale=1, min_width=210, elem_id="sessions-col", visible=False) as sessions_col:
            with gr.Row():
                gr.Markdown("###       Sessions")
            with gr.Row():
                new_session_btn = gr.Button("+ New", variant="secondary", scale=8, min_width=54)
                refresh_sessions_btn = gr.Button("↻", variant="secondary", scale=1, min_width=36)
            sessions_radio = gr.Radio(
                choices=[],
                show_label=False,
                interactive=True,
                elem_id="sessions-radio",
            )

        # MIDDLE: Chat + input area
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="",
                height=460,
                show_label=False,
                elem_id="aoe-chatbot",
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
                param_file = gr.UploadButton(
                    "Upload CSV",
                    file_types=[".csv"],
                    elem_classes=["upload-csv-btn"],
                    visible=False,
                )
                param_df = gr.Dataframe(
                    label="Or edit manually (available when each dimension ≤ 10)",
                    interactive=True,
                    visible=False,
                    elem_id="param-df",
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
                    scale=15,
                    lines=1,        # initial visible height
                    max_lines=5,    # grows until 5 lines
                    autoscroll=True,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1, min_width=80, elem_id="send-btn")

        # RIGHT: Sidebar
        with gr.Column(scale=1, min_width=230):
            gr.Markdown("### MILP Formulation")
            model_display = gr.Image(
                value=None,
                show_label=False,
                buttons=["download", "fullscreen"],
                sources=None,
                elem_classes=["sidebar-card"],
            )

            gr.Markdown("### Confirmed")
            confirmed_display = gr.Markdown(value="*None yet.*", elem_classes=["sidebar-card", "sidebar-scroll"])

            gr.Markdown("### Pending Review")
            unconfirmed_display = gr.Markdown(value="*None yet.*", elem_classes=["sidebar-card", "sidebar-scroll"])

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

    # Populate sessions list on page load
    demo.load(fn=_load_session_list, outputs=[sessions_radio])

    # Auto-scroll the chatbot to the bottom only when the user is already there.
    # If they have scrolled up to read earlier messages, leave their scroll
    # position alone — otherwise every DOM mutation would snap them back down.
    demo.load(
        fn=None,
        inputs=None,
        outputs=None,
        js="""
        () => {
            const rootSel = '#aoe-chatbot';
            const NEAR_BOTTOM_PX = 80;
            const findWrap = (root) => (
                root.querySelector('.bubble-wrap')
                || root.querySelector('[role="log"]')
                || root.querySelector('.message-wrap')
                || root
            );
            const install = () => {
                const root = document.querySelector(rootSel);
                if (!root) { setTimeout(install, 200); return; }
                if (root.dataset.aoeScrollObserver) return;
                root.dataset.aoeScrollObserver = '1';

                let stickToBottom = true;
                const wrap = findWrap(root);

                const updateStick = () => {
                    const w = findWrap(root);
                    const distance = w.scrollHeight - w.scrollTop - w.clientHeight;
                    stickToBottom = distance <= NEAR_BOTTOM_PX;
                };
                wrap.addEventListener('scroll', updateStick, { passive: true });

                new MutationObserver(() => {
                    if (!stickToBottom) return;
                    const w = findWrap(root);
                    w.scrollTop = w.scrollHeight;
                }).observe(root, { childList: true, subtree: true, characterData: true });

                // Start pinned to bottom on first load.
                wrap.scrollTop = wrap.scrollHeight;
            };
            install();
        }
        """,
    )

    # Toggle sessions panel
    toggle_btn.click(
        fn=_toggle_sessions,
        inputs=[sessions_open_state],
        outputs=[sessions_col, sessions_open_state],
    )

    # Auto-resume when user clicks a session
    sessions_radio.change(
        fn=_resume_session,
        inputs=[sessions_radio],
        outputs=ALL_OUTPUTS,
    )

    # Refresh session list button
    refresh_sessions_btn.click(
        fn=_load_session_list,
        outputs=[sessions_radio],
    )

    # New session button (sessions panel)
    new_session_btn.click(
        fn=_reset_session,
        outputs=ALL_OUTPUTS,
    ).then(
        fn=_load_session_list,
        outputs=[sessions_radio],
    )

    evt = gr.on(
        triggers=[send_btn.click, msg_input.submit],
        fn=_process_message,
        inputs=[msg_input, chatbot],
        outputs=ALL_OUTPUTS,
        trigger_mode="once",
    )

    evt.then(
        fn=_refresh_approve_row,
        inputs=[],
        outputs=[approve_row],
    ).then(
        fn=_load_session_list,
        outputs=[sessions_radio],
    ).then(
        fn=None,
        inputs=[],
        outputs=[],
    )

    approve_evt = approve_btn.click(
        fn=_approve,
        inputs=[chatbot],
        outputs=ALL_OUTPUTS,
    )

    approve_evt.then(
        fn=None,
        inputs=[],
        outputs=[],
    )

    changes_btn.click(
        fn=_request_changes,
        inputs=[chatbot],
        outputs=ALL_OUTPUTS,
    )

    data_evt = data_submit_btn.click(
        fn=_submit_param_data,
        inputs=[param_file, param_df, chatbot],
        outputs=ALL_OUTPUTS,
    )

    data_evt.then(
        fn=_refresh_approve_row,
        inputs=[],
        outputs=[approve_row],
    ).then(
        fn=_load_session_list,
        outputs=[sessions_radio],
    )

    # Selecting a CSV in the UploadButton should submit immediately (matches
    # the old behaviour: either accept the data or surface a shape-error).
    upload_evt = param_file.upload(
        fn=_submit_param_data,
        inputs=[param_file, param_df, chatbot],
        outputs=ALL_OUTPUTS,
    )

    upload_evt.then(
        fn=_refresh_approve_row,
        inputs=[],
        outputs=[approve_row],
    ).then(
        fn=_load_session_list,
        outputs=[sessions_radio],
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861, css=custom_css)
