"""
ui/gradio/app.py — Gradio UI for AOE.

Professional dark-mode chat interface for MILP model refinement.
Runs everything in a single process.

Run with:
    python -m ui.gradio.app
"""
"""
ui/gradio/app.py — Gradio UI for AOE.

Professional dark-mode chat interface for MILP model refinement.

Run with:
    python -m ui.gradio.app
"""
import gradio as gr
import sys
from pathlib import Path
from typing import Optional
from middleware.handle import AOEHandle

sys.path.append(str(Path(__file__).parent.parent.parent))

# ============ GLOBAL STATE ============
_aoe_handle   = AOEHandle()
_current_state: Optional[dict] = None

# ============ CORE FUNCTIONS ============

def _build_bot_text(state: dict) -> str:
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
        obj   = milp.get("objective_type", "—")
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


def _process_message(user_message: str, chat_history: list):
    """Handle a normal chat message turn."""
    global _current_state

    if not user_message.strip():
        return "", chat_history, gr.update(visible=False), *(["*Waiting...*"] * 3)

    try:
        _current_state = _aoe_handle.run(user_message, _current_state)
        bot_text       = _build_bot_text(_current_state)

        new_history = []
        for msg in (_current_state or {}).get("history", []):
            new_history.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
        new_history.append({"role": "assistant", "content": bot_text})

        # Show approve row only when open_questions is empty
        questions    = _current_state.get("open_questions") or []
        approve_vis  = gr.update(visible=len(questions) == 0)

        model_info, conf, unconf = _format_sidebar(_current_state)
        return "", new_history, approve_vis, model_info, conf, unconf

    except Exception as e:
        chat_history.append({"role": "assistant", "content": f"**Error:** {str(e)}"})
        return "", chat_history, gr.update(visible=False), "—", "—", "—"


def _approve(chat_history: list):
    """User clicked Approve — set analyser_approved and advance graph."""
    global _current_state
    if _current_state is None:
        return chat_history, gr.update(visible=False), "—", "—", "—"

    _current_state["analyser_approved"] = True
    # Run the graph once more so it can route to code_generator
    _current_state = _aoe_handle.run("__approved__", _current_state)

    chat_history.append({
        "role": "assistant",
        "content": " **Model approved.** Proceeding to code generation…"
    })
    model_info, conf, unconf = _format_sidebar(_current_state)
    return chat_history, gr.update(visible=False), model_info, conf, unconf


def _request_changes(chat_history: list):
    """User clicked Request Changes — hide approve row, prompt for feedback."""
    chat_history.append({
        "role": "assistant",
        "content": "Sure — what would you like to change?"
    })
    return chat_history, gr.update(visible=False)


# ============ CSS ============

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono&display=swap');

:root {
    --aoe-bg:      #0D1B2A;
    --aoe-surface: #152233;
    --aoe-card:    #1C2E40;
    --aoe-b0:      rgba(255,255,255,0.07);
    --aoe-b1:      rgba(255,255,255,0.14);
    --aoe-accent:  #2E7DD1;
    --aoe-asoft:   rgba(46,125,209,0.13);
    --aoe-green:   #1FA876;
    --aoe-t0:      #EFF3F8;
    --aoe-t1:      #8BA3BC;
    --aoe-t2:      #4A6278;
}

body, html { background: var(--aoe-bg) !important; }

.gradio-container {
    background: var(--aoe-bg) !important;
    max-width: 100% !important;
    font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif !important;
}

.contain, .gap, .form, .block, .wrap, .panel {
    background: transparent !important;
    border-color: var(--aoe-b0) !important;
}

body, .gradio-container,
p, span, label, h1, h2, h3, h4, h5, li, td, th {
    color: var(--aoe-t0) !important;
    font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif !important;
}

.aoe-title {
    font-size: 11px !important;
    font-weight: 700 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    text-align: center !important;
    color: var(--aoe-t0) !important;
    padding: 12px 0 !important;
    border-bottom: 1px solid var(--aoe-b1) !important;
    margin-bottom: 4px !important;
}

.chatbot, [data-testid="chatbot"] {
    background: var(--aoe-surface) !important;
    border: 1px solid var(--aoe-b1) !important;
    border-radius: 10px !important;
}

.message { border-radius: 8px !important; padding: 10px 14px !important; }
.message.user {
    background: var(--aoe-asoft) !important;
    border: 1px solid rgba(46,125,209,0.3) !important;
}
.message.bot, .message.assistant {
    background: var(--aoe-card) !important;
    border: 1px solid var(--aoe-b0) !important;
    color: var(--aoe-t0) !important;
}
.avatar-container { background: var(--aoe-card) !important; }
.message.bot strong, .message.assistant strong {
    color: var(--aoe-t0) !important;
}

textarea, input[type="text"] {
    background: var(--aoe-card) !important;
    color: var(--aoe-t0) !important;
    border: 1px solid var(--aoe-b1) !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    padding: 10px 14px !important;
}
textarea:focus, input:focus {
    border-color: var(--aoe-accent) !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(46,125,209,0.18) !important;
}
textarea::placeholder, input::placeholder { color: var(--aoe-t2) !important; }

.approve-row p, .approve-row span, .approve-row strong {
    color: var(--aoe-t0) !important;
}

button {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    border-radius: 7px !important;
    cursor: pointer !important;
}
button.primary, [class*="primary"]:not(.message) {
    background: var(--aoe-accent) !important;
    color: #fff !important;
    border: none !important;
}
button.primary:hover { opacity: 0.85 !important; }
button.secondary, [class*="secondary"]:not(.message) {
    background: transparent !important;
    color: var(--aoe-t1) !important;
    border: 1px solid var(--aoe-b1) !important;
}
button.secondary:hover { background: var(--aoe-card) !important; color: var(--aoe-t0) !important; }

/* Approve row */
.approve-row {
    background: var(--aoe-card) !important;
    border: 1px solid var(--aoe-b1) !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
    margin-top: 6px !important;
}
.approve-label {
    font-size: 12px !important;
    color: var(--aoe-t1) !important;
    margin-bottom: 8px !important;
}
.approve-btn {
    background: var(--aoe-green) !important;
    color: #fff !important;
    border: none !important;
}
.approve-btn:hover { opacity: 0.85 !important; }
.changes-btn {
    background: transparent !important;
    color: var(--aoe-t1) !important;
    border: 1px solid var(--aoe-b1) !important;
}

.sidebar-card {
    background: var(--aoe-card) !important;
    border: 1px solid var(--aoe-b0) !important;
    border-radius: 8px !important;
    padding: 12px 14px !important;
    margin-bottom: 2px !important;
}
.sidebar-card p, .sidebar-card li { color: var(--aoe-t1) !important; font-size: 12px !important; }
.sidebar-card code {
    color: var(--aoe-accent) !important;
    background: var(--aoe-asoft) !important;
    border-radius: 3px !important;
    padding: 1px 5px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
}
.sidebar-card table { width: 100% !important; border-collapse: collapse !important; }
.sidebar-card th {
    font-size: 10px !important; letter-spacing: 0.07em !important;
    text-transform: uppercase !important; color: var(--aoe-t2) !important;
    border-bottom: 1px solid var(--aoe-b1) !important; padding: 4px 6px !important;
}
.sidebar-card td {
    font-size: 12px !important; color: var(--aoe-t1) !important;
    padding: 5px 6px !important; border-bottom: 1px solid var(--aoe-b0) !important;
}

h3 {
    font-size: 10px !important; font-weight: 700 !important;
    letter-spacing: 0.1em !important; text-transform: uppercase !important;
    color: var(--aoe-t2) !important; margin: 14px 0 4px !important;
}

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--aoe-bg); }
::-webkit-scrollbar-thumb { background: var(--aoe-b1); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: var(--aoe-accent); }

footer, .footer { display: none !important; }
"""

# ============ GRADIO UI ============

with gr.Blocks(title="AOE — Automated Optimization Engineer") as demo:

    gr.Markdown("<p class='aoe-title'>AOE &nbsp;/&nbsp; Automated Optimization Engineer</p>")

    with gr.Row(equal_height=True):

        # LEFT: Chat
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="", height=460, show_label=False)

            # Approve / Request Changes row — hidden until open_questions is empty
            with gr.Row(visible=False, elem_classes=["approve-row"]) as approve_row:
                gr.Markdown(
                    "**Does this model look correct?**",
                    elem_classes=["approve-label"],
                )
                approve_btn  = gr.Button("✔ Approve",          elem_classes=["approve-btn"],  scale=1)
                changes_btn  = gr.Button("✎ Request Changes",  elem_classes=["changes-btn"],  scale=1)

            # Normal input row
            with gr.Row():
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

    # ── Bindings ──────────────────────────────────────────────────────
    send_outputs = [msg_input, chatbot, approve_row, model_display, confirmed_display, unconfirmed_display]

    send_btn.click(fn=_process_message, inputs=[msg_input, chatbot], outputs=send_outputs)
    msg_input.submit(fn=_process_message, inputs=[msg_input, chatbot], outputs=send_outputs)

    approve_btn.click(
        fn=_approve,
        inputs=[chatbot],
        outputs=[chatbot, approve_row, model_display, confirmed_display, unconfirmed_display],
    )

    changes_btn.click(
        fn=_request_changes,
        inputs=[chatbot],
        outputs=[chatbot, approve_row],
    )

    def reset_session():
        global _current_state
        _current_state = None
        return [], "", gr.update(visible=False), "*Waiting for input...*", "*None yet.*", "*None yet.*"

    reset_btn.click(
        fn=reset_session,
        outputs=[chatbot, msg_input, approve_row, model_display, confirmed_display, unconfirmed_display],
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, css=custom_css)
