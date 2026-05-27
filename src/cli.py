"""Interactive command-line interface for the agent.

Streams the agent's reasoning steps (router decision, tool calls, tool
results) to the terminal as they happen, then prints the final answer.
"""

from typing import Any, Optional

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from src.agent import FALLBACK_MESSAGE, MAX_ITERATIONS, graph


# ---------------------------------------------------------------------------
# ANSI colors — make the trace readable in the terminal
# ---------------------------------------------------------------------------


class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    BLUE = "\033[94m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    GRAY = "\033[90m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    RED = "\033[91m"


# ---------------------------------------------------------------------------
# Helpers for displaying stream events
# ---------------------------------------------------------------------------


def _truncate(text: str, limit: int = 300) -> str:
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f" ... ({len(text) - limit} more chars)"


def _format_args(args: dict) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in args.items())


def _print_router(route: str, reason: str) -> None:
    print(f"{C.BLUE}[Router]{C.RESET} {C.BOLD}{route}{C.RESET} — {C.DIM}{reason}{C.RESET}")


def _print_tool_call(name: str, args: dict) -> None:
    print(f"{C.YELLOW}[Tool call]{C.RESET} {C.BOLD}{name}{C.RESET}({_format_args(args)})")


def _print_tool_result(name: str, content: str) -> None:
    print(f"{C.GRAY}[Result of {name}]{C.RESET} {_truncate(content)}")


def _print_thinking(content: str) -> None:
    """The agent sometimes emits intermediate reasoning text alongside tool calls."""
    print(f"{C.CYAN}[Thinking]{C.RESET} {_truncate(content, 400)}")


def _print_final(content: str) -> None:
    print()
    print(f"{C.GREEN}{C.BOLD}🤖 {content}{C.RESET}")
    print()


def _print_decline(content: str) -> None:
    print()
    print(f"{C.MAGENTA}🤖 {content}{C.RESET}")
    print()


# ---------------------------------------------------------------------------
# Event handler — extracts router decisions, tool calls, and final answers
# from graph.stream() events
# ---------------------------------------------------------------------------


def _handle_update(
    update: dict, in_subgraph: bool, seen_message_ids: set
) -> Optional[str]:
    """Process one update dict from graph.stream().

    Args:
        update: The {node_name: state_update} dict yielded by graph.stream().
        in_subgraph: True when the event came from inside a subgraph (i.e.
            the namespace tuple was non-empty). We use this to skip the
            parent graph's rollup of the ReAct subgraph, which would
            otherwise duplicate every event we already streamed from inside.
        seen_message_ids: A set used to deduplicate messages across events.

    Returns:
        The final answer string when the run finishes, otherwise None.
    """
    final_answer: Optional[str] = None

    for node_name, payload in update.items():
        if not isinstance(payload, dict):
            continue

        # The parent graph sees a single rolled-up update for the "agent"
        # node containing every message the ReAct subgraph appended. Since
        # we already streamed those messages from inside the subgraph, skip.
        if node_name == "agent" and not in_subgraph:
            continue

        # 1. Router decision (top-level only)
        if node_name == "router" and "route" in payload:
            _print_router(payload["route"], payload.get("route_reason", ""))
            continue

        # 2. Decline node
        if node_name == "decline":
            for msg in payload.get("messages", []):
                if isinstance(msg, AIMessage) and id(msg) not in seen_message_ids:
                    seen_message_ids.add(id(msg))
                    _print_decline(msg.content)
                    final_answer = msg.content
            continue

        # 3. ReAct agent (subgraph) - tool calls, intermediate text, final answer
        for msg in payload.get("messages", []):
            if id(msg) in seen_message_ids:
                continue
            seen_message_ids.add(id(msg))
            if isinstance(msg, AIMessage):
                tool_calls = getattr(msg, "tool_calls", None) or []
                if tool_calls:
                    if msg.content:
                        _print_thinking(msg.content)
                    for call in tool_calls:
                        _print_tool_call(call["name"], call.get("args", {}))
                else:
                    _print_final(msg.content)
                    final_answer = msg.content
            elif isinstance(msg, ToolMessage):
                _print_tool_result(msg.name or "tool", msg.content)

    return final_answer


# ---------------------------------------------------------------------------
# Main CLI loop
# ---------------------------------------------------------------------------


WELCOME = f"""{C.BOLD}{C.CYAN}Customer Service Data Analyst Agent{C.RESET}
Ask me anything about the Bitext customer-support dataset:
  - "What categories exist?"
  - "How many refund requests?"
  - "Summarize the FEEDBACK category."
Commands: {C.DIM}/help, /exit, /quit{C.RESET}
"""


HELP_TEXT = f"""{C.BOLD}Commands:{C.RESET}
  /help          — show this message
  /exit, /quit   — leave the chat
"""


def _handle_command(cmd: str) -> bool:
    """Returns True if the user wants to exit."""
    cmd = cmd.strip().lower()
    if cmd in ("/exit", "/quit"):
        print(f"{C.DIM}Goodbye.{C.RESET}")
        return True
    if cmd == "/help":
        print(HELP_TEXT)
        return False
    print(f"{C.RED}Unknown command: {cmd}. Try /help.{C.RESET}")
    return False


def _run_one_turn(question: str, session_id: str) -> None:
    """Stream the graph for one user turn, printing events as they arrive.

    Args:
        question: The user's question.
        session_id: Used as the LangGraph `thread_id` so the checkpointer
            (Task 2a) can persist + restore conversation state per session.
    """
    final_answer: Optional[str] = None
    seen_message_ids: set = set()
    try:
        stream = graph.stream(
            {"messages": [HumanMessage(content=question)]},
            config={
                "configurable": {"thread_id": session_id},
                "recursion_limit": MAX_ITERATIONS * 2 + 4,
            },
            stream_mode="updates",
            subgraphs=True,
        )
        for event in stream:
            # With subgraphs=True, events come as (namespace_tuple, update_dict).
            # A non-empty namespace means the event originated inside a subgraph
            # (the ReAct agent); empty namespace is the top-level graph.
            if isinstance(event, tuple) and len(event) == 2:
                namespace, update = event
                in_subgraph = bool(namespace)
            else:
                update = event
                in_subgraph = False
            answer = _handle_update(update, in_subgraph, seen_message_ids)
            if answer is not None:
                final_answer = answer
    except GraphRecursionError:
        _print_final(FALLBACK_MESSAGE)
        return

    if final_answer is None:
        print(f"{C.RED}(No final answer was produced.){C.RESET}")


def run_cli(session_id: Optional[str] = None) -> None:
    """Run the interactive REPL.

    Args:
        session_id: The session identifier (Task 2). If omitted, falls back to
            'default' — memory is always on, the same 'default' session is
            reused across runs that don't specify --session.
    """
    if not session_id:
        session_id = "default"
    print(WELCOME)
    print(f"{C.DIM}Session: {session_id}{C.RESET}")
    print()

    while True:
        try:
            question = input(f"{C.BOLD}You:{C.RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            print(f"{C.DIM}Goodbye.{C.RESET}")
            break

        if not question:
            continue

        if question.startswith("/"):
            if _handle_command(question):
                break
            continue

        _run_one_turn(question, session_id)
