"""Streamlit chat UI for the Bitext data-analyst agent (Bonus A).

A web wrapper around the same LangGraph agent the CLI uses. The full
reasoning trace (router decisions, tool calls, tool results, final
answer) is rendered in the browser, and a sidebar lets the user
switch between session IDs to load different conversation histories.

Run with:
    streamlit run streamlit_app.py
"""

import json
import warnings

warnings.filterwarnings("ignore")

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from src.agent import FALLBACK_MESSAGE, MAX_ITERATIONS, graph


# ---------------------------------------------------------------------------
# Page config + helpers
# ---------------------------------------------------------------------------


st.set_page_config(
    page_title="Customer Service Data Analyst Agent",
    page_icon="🤖",
    layout="wide",
)


def _strip_thinking_tags(text: str) -> str:
    if "</think>" in text:
        return text.split("</think>", 1)[-1].strip()
    return text.strip()


def _truncate(text: str, limit: int = 500) -> str:
    text = str(text)
    return text if len(text) <= limit else text[:limit].rstrip() + f"  …(+{len(text) - limit} chars)"


def _format_args(args: dict) -> str:
    parts = []
    for k, v in args.items():
        if v is None or v == "null":
            continue
        parts.append(f"{k}={v!r}")
    return ", ".join(parts)


def _try_pretty_json(text: str) -> str:
    """If the text is valid JSON, pretty-print it; otherwise return as-is."""
    try:
        return json.dumps(json.loads(text), indent=2, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        return text


def _load_history_from_checkpointer(thread_id: str) -> list[dict]:
    """Load the prior conversation for `thread_id` from LangGraph's checkpointer.

    Returns a flat list of {"role", "content"} dicts ready to be replayed by
    st.chat_message. Tool calls and tool results are filtered out — only
    user messages and final assistant answers are shown in the chat.
    """
    try:
        state = graph.get_state(config={"configurable": {"thread_id": thread_id}})
    except Exception:
        return []

    if not state or not state.values:
        return []

    history: list[dict] = []
    for msg in state.values.get("messages", []) or []:
        if isinstance(msg, HumanMessage) and msg.content:
            history.append({"role": "user", "content": str(msg.content)})
        elif isinstance(msg, AIMessage):
            tool_calls = getattr(msg, "tool_calls", None) or []
            if tool_calls:
                continue  # intermediate tool-calling turn; skip in display
            content = _strip_thinking_tags(str(msg.content))
            if content:
                history.append({"role": "assistant", "content": content})
    return history


# ---------------------------------------------------------------------------
# Stream-event handler — mirrors src/cli.py but writes to a Streamlit
# status box instead of stdout.
# ---------------------------------------------------------------------------


def _handle_update(update: dict, in_subgraph: bool, seen_ids: set, status) -> str | None:
    """Process one update dict from graph.stream(). Returns final answer or None."""
    final_answer: str | None = None

    for node_name, payload in update.items():
        if not isinstance(payload, dict):
            continue

        # Skip the parent graph's rollup of the agent subgraph — we already
        # saw those messages individually from inside the subgraph.
        if node_name == "agent" and not in_subgraph:
            continue

        # 1. Router decision
        if node_name == "router" and "route" in payload:
            route = payload["route"]
            reason = payload.get("route_reason", "")
            status.markdown(f"🔀 **Router** → `{route}`  \n_{reason}_")
            continue

        # 2. Decline node
        if node_name == "decline":
            for msg in payload.get("messages", []):
                if isinstance(msg, AIMessage) and id(msg) not in seen_ids:
                    seen_ids.add(id(msg))
                    final_answer = msg.content
            continue

        # 3. ReAct subgraph — tool calls, tool results, final answer
        for msg in payload.get("messages", []):
            if id(msg) in seen_ids:
                continue
            seen_ids.add(id(msg))

            if isinstance(msg, AIMessage):
                tool_calls = getattr(msg, "tool_calls", None) or []
                if tool_calls:
                    for call in tool_calls:
                        args_str = _format_args(call.get("args", {}))
                        status.markdown(f"🔧 **Tool call**: `{call['name']}({args_str})`")
                else:
                    final_answer = _strip_thinking_tags(str(msg.content))

            elif isinstance(msg, ToolMessage):
                tool_name = msg.name or "tool"
                content = _truncate(_try_pretty_json(str(msg.content)))
                status.markdown(f"📊 **Result** from `{tool_name}`:")
                status.code(content, language="json")

    return final_answer


# ---------------------------------------------------------------------------
# Sidebar — session control
# ---------------------------------------------------------------------------


with st.sidebar:
    st.header("Session")
    session_id = st.text_input(
        "Session ID",
        value="default",
        help=(
            "Conversations are saved per session. Change this to switch to "
            "or resume a different conversation."
        ),
    )
    st.caption(f"Current thread: `{session_id or 'default'}`")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------


st.title("🤖 Customer Service Data Analyst Agent")
st.caption(
    "Ask questions about the Bitext customer-support dataset. "
    "Examples: *'What categories exist?'*, *'How many refund requests?'*, "
    "*'Summarize the FEEDBACK category.'*"
)


# ---------------------------------------------------------------------------
# Chat state — bound to the selected session
# ---------------------------------------------------------------------------


effective_session = session_id or "default"


# When the session changes (or on first load), discard the in-browser chat
# log and repopulate it from the LangGraph checkpointer so the user "resumes"
# the right conversation. Stored under a different key per session id so
# switching back and forth doesn't lose state mid-typing.
if (
    "current_session" not in st.session_state
    or st.session_state.current_session != effective_session
):
    loaded = _load_history_from_checkpointer(effective_session)
    st.session_state.messages = loaded
    st.session_state.current_session = effective_session
    if loaded:
        st.toast(
            f"Resumed session `{effective_session}` ({len(loaded)} prior message{'s' if len(loaded) != 1 else ''})",
            icon="📂",
        )


# Replay the chat history on every rerun. Only final answers are stored —
# the live reasoning trace is shown once, while the agent is running.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ---------------------------------------------------------------------------
# Chat input + agent call (streamed with reasoning trace)
# ---------------------------------------------------------------------------


if prompt := st.chat_input("Ask about the data..."):
    # 1. Show the user's message.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Stream the agent, showing each reasoning step inside a status box.
    # (effective_session was resolved above when the session-change block ran.)
    seen_ids: set = set()
    final_answer: str | None = None

    with st.chat_message("assistant"):
        try:
            with st.status("🧠 Thinking…", expanded=True) as status:
                stream = graph.stream(
                    {"messages": [HumanMessage(content=prompt)]},
                    config={
                        "configurable": {"thread_id": effective_session},
                        "recursion_limit": MAX_ITERATIONS * 2 + 4,
                    },
                    stream_mode="updates",
                    subgraphs=True,
                )
                for event in stream:
                    if isinstance(event, tuple) and len(event) == 2:
                        namespace, update = event
                        in_subgraph = bool(namespace)
                    else:
                        update = event
                        in_subgraph = False
                    ans = _handle_update(update, in_subgraph, seen_ids, status)
                    if ans is not None:
                        final_answer = ans

                status.update(label="✅ Reasoning complete", state="complete", expanded=False)
        except GraphRecursionError:
            final_answer = FALLBACK_MESSAGE

        # 3. Final answer (always visible, outside the collapsible status box).
        if final_answer:
            st.markdown(final_answer)
        else:
            st.warning("The agent did not produce a final answer.")

    st.session_state.messages.append(
        {"role": "assistant", "content": final_answer or "(no answer)"}
    )
