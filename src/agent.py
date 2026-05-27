"""LangGraph wiring: Router -> (Decline | ReAct Agent) -> END.

This module assembles the full agent graph from the smaller pieces:
    - src.router.route_query   (classifies the user query)
    - src.tools.ALL_TOOLS      (the tools the agent can call)
    - src.config.get_llm       (creates the Nebius-backed LLM)

The graph itself is small (3 nodes): a router, a polite decline node for
out-of-scope queries, and the prebuilt ReAct agent from LangGraph for the
structured / unstructured branches.
"""

import sqlite3
from operator import add
from pathlib import Path
from typing import Annotated, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import create_react_agent
from typing_extensions import TypedDict

from src.config import AGENT_MODEL, MAX_ITERATIONS, get_llm
from src.memory import load_profile, update_profile_node
from src.router import route_query
from src.tools import ALL_TOOLS


# ---------------------------------------------------------------------------
# Checkpointer (Task 2a: persistent episodic memory across restarts)
# ---------------------------------------------------------------------------

CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "checkpoints.sqlite"

# Persistent SQLite connection lives for the lifetime of the process.
# check_same_thread=False because LangGraph may touch the connection from
# its internal worker threads when streaming.
_checkpoint_conn = sqlite3.connect(str(CHECKPOINT_PATH), check_same_thread=False)
_checkpointer = SqliteSaver(_checkpoint_conn)
_checkpointer.setup()  # idempotent — creates the tables on first run.


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    """The state object that flows through the graph.

    - messages: chat history, accumulated by LangGraph's `add_messages` reducer.
    - route: the router's classification, set by router_node and read by the
      conditional edge to decide which node runs next.
    - user_id: the session/user identifier, copied from the runtime config's
      thread_id by router_node. Used by the profile-aware prompt and the
      summary node (Task 2b).
    - remaining_steps: required by LangGraph's create_react_agent when we
      pass it state_schema=AgentState; tracks how many steps the agent has
      left before the recursion limit kicks in.
    """

    messages: Annotated[list, add_messages]
    route: str
    route_reason: str
    user_id: str
    remaining_steps: RemainingSteps


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


AGENT_SYSTEM_PROMPT = """You are a data analyst agent for the Bitext customer-support dataset.

THE DATASET
- 26,872 rows of customer-support interactions.
- Each row has: 'instruction' (what the customer wrote), 'response' (what the
  agent wrote), 'category' (high-level topic), and 'intent' (specific goal).
- 11 categories: ACCOUNT, CANCEL, CONTACT, DELIVERY, FEEDBACK, INVOICE,
  ORDER, PAYMENT, REFUND, SHIPPING, SUBSCRIPTION.
- 27 intents (e.g. cancel_order, get_refund, complaint, track_order).

YOUR TOOLS
- list_categories: get the full list of category names.
- list_intents: get the list of intent names, optionally filtered to one category.
- count_rows: count rows, optionally filtered by category and/or intent.
- get_examples: random sample of rows for "show me examples" questions.
- intent_distribution: intent -> count breakdown for a given category.
- get_texts_for_summary: fetch a batch of rows for YOU to read and summarize.

WHEN TO USE WHICH TOOL
- "How many X" -> count_rows. If the user uses informal wording
  (e.g. "people wanting their money back"), first call list_intents to
  discover the matching intent name (e.g. 'get_refund'), then count.
- "Show me N examples of X" -> get_examples.
- "Summarize X" or "How do agents respond to X" -> get_texts_for_summary,
  THEN YOU read the returned texts and write a natural-language summary.
- "Distribution of intents in X" -> intent_distribution.
- For names you're unsure about (category or intent), call list_categories
  or list_intents first to discover the exact name.

GENERAL RULES
- Always cite actual numbers, category names, and intent names from tool results.
- Be concise but complete: a number, a short interpretation, that's enough.
- If you need information that the tools can't provide, say so honestly.

STOPPING RULES (very important)
- One call per tool is usually enough. If you already have the data you need,
  STOP calling tools and write your final answer.
- NEVER call the same tool twice with the same arguments — the data will not
  change in a meaningful way.
- After a successful tool result, your next message must be the FINAL ANSWER
  to the user, unless you genuinely need to chain to a DIFFERENT tool
  (e.g., list_intents then count_rows).
- If a tool returns N rows of examples, those ARE the examples. Present them
  to the user in your final answer — do not ask the tool for more.
- When you don't want to filter by an optional argument, OMIT it entirely.
  Do not pass the string 'null' or 'None' — omit the argument.

USING CONVERSATION HISTORY (multi-turn)
- The conversation history above is yours to use. If a question refers to
  prior turns ("3 more", "what about X", "the last two", "those examples"),
  resolve the reference from the history before acting.
- If the user asks for arithmetic over numbers you already reported in
  earlier turns (e.g. "total of the last two", "sum them"), DO THE
  ARITHMETIC YOURSELF from the prior answers — do NOT call count_rows
  again, since you already have the numbers."""


DECLINE_MESSAGE = (
    "I'm a customer-service data-analyst agent and can only answer questions "
    "about the Bitext customer-support dataset (categories, intents, examples, "
    "and summaries from the data). Please ask me something about the dataset."
)


FALLBACK_MESSAGE = (
    f"I couldn't reach a final answer within the iteration limit "
    f"({MAX_ITERATIONS} steps). Could you rephrase your question or "
    "break it into smaller parts?"
)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def router_node(state: AgentState, config: RunnableConfig) -> dict:
    """Classify the latest user message and write the decision into state.

    Also pulls the session/user identifier from the runtime config's
    thread_id and stores it on the state so downstream nodes can use it
    (e.g. the profile-aware prompt and the summary node, Task 2b).

    Passes the prior conversation (everything except the latest message)
    to the router so it can correctly classify follow-up questions like
    "show me 3 more" or "total of the last two".
    """
    messages = state["messages"]
    last_user_message = messages[-1].content
    prior_history = messages[:-1]
    decision = route_query(last_user_message, history=prior_history)
    user_id = (config.get("configurable") or {}).get("thread_id") or "default"
    return {
        "route": decision.category,
        "route_reason": decision.reason,
        "user_id": user_id,
    }


def decline_node(state: AgentState) -> dict:
    """Polite refusal for out-of-scope queries."""
    return {"messages": [AIMessage(content=DECLINE_MESSAGE)]}


def decide_after_router(state: AgentState) -> Literal["decline", "agent"]:
    """Edge function: pick the next node based on the router's decision."""
    if state["route"] == "out_of_scope":
        return "decline"
    return "agent"


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------


def _build_agent_prompt(state: AgentState) -> list:
    """Build the agent's prompt dynamically, injecting the user's profile.

    The base AGENT_SYSTEM_PROMPT is constant, but on every turn we append
    the freshly-loaded profile for the current user_id. This means the
    agent always sees the LATEST profile content (the summary node may
    have updated it on the previous turn).
    """
    user_id = state.get("user_id") or "default"
    profile = load_profile(user_id)
    system = AGENT_SYSTEM_PROMPT
    if profile:
        system += (
            "\n\n---\n"
            "USER PROFILE (long-term facts about the person you're talking to;\n"
            "use them to personalize and to answer questions like\n"
            "'what do you remember about me?'; do not treat them as dataset content):\n"
            f"{profile}"
        )
    return [SystemMessage(content=system)] + list(state.get("messages") or [])


# The ReAct agent is itself a compiled LangGraph. We use the prebuilt one
# from langgraph.prebuilt — it handles the Think / Act / Observe loop and
# tool-call routing for us, so we don't have to reinvent it. The `prompt`
# is a callable so the user profile can be injected dynamically per turn.
_react_agent = create_react_agent(
    model=get_llm(AGENT_MODEL),
    tools=ALL_TOOLS,
    state_schema=AgentState,
    prompt=_build_agent_prompt,
)


def _build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("router", router_node)
    builder.add_node("decline", decline_node)
    builder.add_node("agent", _react_agent)
    builder.add_node("summary", update_profile_node)

    builder.set_entry_point("router")
    builder.add_conditional_edges(
        "router",
        decide_after_router,
        {"decline": "decline", "agent": "agent"},
    )
    # Both paths (decline and agent) go through the summary node so the
    # per-user profile (Task 2b) gets a chance to update after every turn.
    builder.add_edge("decline", "summary")
    builder.add_edge("agent", "summary")
    builder.add_edge("summary", END)

    return builder.compile(checkpointer=_checkpointer)


graph = _build_graph()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_agent(question: str, recursion_limit: int | None = None) -> str:
    """Run the agent on a single question and return the final answer text.

    Args:
        question: The user's question, in natural language.
        recursion_limit: Optional override for the LangGraph recursion limit.
            Each ReAct iteration uses 2 super-steps (agent + tool), so we
            default to MAX_ITERATIONS * 2 + 4 to also cover the router and
            the agent's final reply.

    Returns:
        The final answer string. If the agent runs out of iterations, a
        graceful fallback message is returned instead.
    """
    limit = recursion_limit if recursion_limit is not None else MAX_ITERATIONS * 2 + 4
    try:
        result = graph.invoke(
            {"messages": [HumanMessage(content=question)]},
            config={"recursion_limit": limit},
        )
        return result["messages"][-1].content
    except GraphRecursionError:
        return FALLBACK_MESSAGE
