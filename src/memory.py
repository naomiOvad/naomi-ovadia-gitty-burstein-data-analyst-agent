"""User-profile storage and the summary node (Task 2b).

Each user has a profile file at `context/<user_id>.md`. The profile holds
DURABLE facts about the user (name, role, interests, preferences) — it is
NOT a replay of past messages. A "summary node" runs after every turn,
reads the latest exchange + current profile, and asks a small LLM whether
anything new should be added. The profile is then injected into the agent's
system prompt on subsequent turns (see src/agent.py).
"""

from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from src.config import ROUTER_MODEL, get_llm


PROFILE_DIR = Path(__file__).resolve().parent.parent / "context"


PROFILE_UPDATE_SYSTEM_PROMPT = """You maintain a long-term profile of a single user across many conversations.

Given the CURRENT profile and the LATEST exchange (the user's message and the agent's reply), return the UPDATED profile.

Include ONLY durable facts about the user, such as:
- Name, role, profession, company
- Long-term interests or recurring topics they care about
- Stated preferences (e.g., "prefers concise answers", "likes bullet points")
- Goals or projects the user is working on
- Anything they explicitly ask you to remember

EXCLUDE:
- Specific data questions the user asked (e.g., "asked how many refunds") — this is NOT a replay of past messages.
- Answers the assistant gave (e.g., "told them there are 997 refunds").
- Transient or one-off state ("currently exploring X").
- Small talk and trivialities.

Rules:
- Keep the profile concise: Markdown bullet list, one short line per fact.
- If the new exchange contains no durable new facts, return the CURRENT profile UNCHANGED (same text).
- If a new fact contradicts or refines an existing fact, REPLACE the old one.
- NEVER include chain-of-thought tags ("<think>...</think>") or any narration about what you did.

Output ONLY the final profile content. No preamble. No explanation."""


# Use the small/fast router model for profile updates: simple summarization
# doesn't need the heavier agent model and would cost too much per turn.
_summary_llm = get_llm(ROUTER_MODEL, temperature=0.0)


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def profile_path(user_id: str) -> Path:
    """Return the filesystem path of the profile file for a given user."""
    return PROFILE_DIR / f"{user_id}.md"


def load_profile(user_id: str) -> str:
    """Return the user's profile as a markdown string. Empty string if none."""
    path = profile_path(user_id)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def save_profile(user_id: str, content: str) -> None:
    """Overwrite the user's profile file with the given markdown content."""
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    profile_path(user_id).write_text(content.strip() + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_thinking_tags(text: str) -> str:
    """Qwen3 models emit '<think>...</think>' chain-of-thought blocks.

    Strip them so we don't pollute the profile file.
    """
    if "</think>" in text:
        return text.split("</think>", 1)[-1].strip()
    return text.strip()


def _format_recent_exchange(messages: list, max_messages: int = 4) -> str:
    """Format the most recent user/assistant turns as a short transcript.

    Only the textual content of HumanMessage and AIMessage is included;
    tool calls and tool results are skipped (they are transient by nature).
    """
    relevant: list[str] = []
    for msg in messages[-(max_messages * 2):]:
        if isinstance(msg, HumanMessage) and msg.content:
            relevant.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage) and msg.content:
            content = _strip_thinking_tags(str(msg.content))
            if content:
                relevant.append(f"Assistant: {content}")
    return "\n".join(relevant[-(max_messages * 2):])


# ---------------------------------------------------------------------------
# Graph node
# ---------------------------------------------------------------------------


def update_profile_node(state: dict, config: RunnableConfig) -> dict:
    """Graph node: distill durable facts from the latest exchange.

    Reads the user's current profile and the most recent messages, asks the
    summary LLM whether the profile should be updated, and overwrites the
    profile file on disk if anything changed. Does not mutate graph state.
    """
    user_id = state.get("user_id") or (
        (config.get("configurable") or {}).get("thread_id") or "default"
    )

    messages = state.get("messages") or []
    exchange = _format_recent_exchange(messages)
    if not exchange:
        return {}

    current_profile = load_profile(user_id)

    user_msg = (
        f"CURRENT PROFILE:\n{current_profile or '(empty — no profile yet)'}\n\n"
        f"LATEST EXCHANGE:\n{exchange}\n\n"
        "Now output the updated profile (or the unchanged profile if nothing durable is new):"
    )

    try:
        response = _summary_llm.invoke(
            [
                SystemMessage(content=PROFILE_UPDATE_SYSTEM_PROMPT),
                HumanMessage(content=user_msg),
            ]
        )
    except Exception:
        # Never let a profile-update failure break the agent turn.
        return {}

    new_profile = _strip_thinking_tags(str(response.content)).strip()
    if not new_profile:
        return {}

    # Only write to disk if the content actually changed.
    if new_profile != current_profile:
        save_profile(user_id, new_profile)

    return {}
