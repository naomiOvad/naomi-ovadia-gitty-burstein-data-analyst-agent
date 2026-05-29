"""Query router: classifies an incoming user query into one of three buckets.

The router is the first node in the agent graph. It uses the small, fast
ROUTER_MODEL to decide whether the question is:
    - structured: can be answered with concrete data operations (count, filter, list)
    - unstructured: requires reading text and summarizing
    - out_of_scope: unrelated to the dataset, should be politely declined

Out-of-scope queries must be detected here so the agent never answers them
from general LLM knowledge.
"""

from typing import List, Literal, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.config import ROUTER_MODEL, get_llm


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


RouteCategory = Literal["structured", "unstructured", "out_of_scope"]


class RouteDecision(BaseModel):
    """The router's classification of a user query."""

    category: RouteCategory = Field(
        description=(
            "The query type: 'structured' for concrete data queries "
            "(counts, lists, examples), 'unstructured' for open-ended "
            "summarization questions, 'out_of_scope' for anything "
            "unrelated to the customer service dataset."
        ),
    )
    reason: str = Field(
        description="One short sentence explaining the classification.",
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


ROUTER_SYSTEM_PROMPT = """You are a query classifier for a customer-service data-analyst agent.

The agent answers questions about the Bitext customer-support dataset.
The dataset contains ~27,000 rows. Each row is a single customer query
("instruction") paired with an agent response ("response"), labeled with
a high-level "category" (e.g. ACCOUNT, REFUND, SHIPPING) and a specific
"intent" (e.g. cancel_order, get_refund, complaint).

Your job: classify each user query into exactly ONE of three categories.

1. structured
   The question has a concrete, data-driven answer that can be computed
   by filtering, counting, listing, or sampling rows from the dataset.
   Indicators: "how many", "what categories", "show me N examples",
   "list", "distribution".
   Examples:
     - "What categories exist in the dataset?"
     - "How many refund requests did we get?"
     - "Show me 3 examples from the SHIPPING category."
     - "What is the distribution of intents in the ACCOUNT category?"
     - "Show me examples of people wanting their money back."

2. unstructured
   The question is open-ended and requires the LLM to read text content
   from the dataset and summarize or characterize it in natural language.
   Indicators: "summarize", "how do agents respond", "what is the tone",
   "describe", "what patterns".
   Examples:
     - "Summarize the FEEDBACK category."
     - "How do customer service representatives typically respond to cancellation requests?"
     - "What's the tone of customer complaints?"

3. out_of_scope
   The question is unrelated to the customer-service dataset. This includes
   general world knowledge, opinions, creative writing, software
   recommendations, or anything that cannot be answered from the dataset.
   Examples:
     - "Who won the 2024 Champions League?"
     - "Write me a poem about customer service."
     - "What's the best CRM software for handling complaints?"
     - "Who is the president of France?"

Important rules:
- A question is in-scope ONLY if answering it requires looking at the dataset.
- General questions about customer service as a topic (e.g. "best CRM
  software", "tips for handling complaints") are out_of_scope — they don't
  ask about the data we have.
- If the question references our data ("our dataset", "the categories",
  "people in our data wanting refunds"), prefer structured or unstructured.

Follow-up questions (very important):
- The user may be in the middle of a multi-turn conversation. The latest
  question can be a follow-up that only makes sense given the prior turns
  (e.g. "show me 3 more", "what about refunds?", "what is the total of the
  last two?", "and SHIPPING?").
- When the recent conversation is provided below, USE IT to interpret the
  latest question. Such follow-ups are usually 'structured' (they ultimately
  refer to data: more examples, another count, arithmetic over earlier
  counts) — NOT 'out_of_scope'.
- A question that asks for arithmetic over earlier numerical answers
  (e.g. "total of the last two", "sum them") is 'structured'.

Personal / profile-related messages (also important):
- The agent keeps a long-term profile of each user. The following are
  IN-SCOPE (NOT out_of_scope), because the agent itself must respond to
  them, even if they don't query the dataset:
  - The user introducing themselves ("Hi, I'm Naomi", "My name is X",
    "I work as Y", "I'm interested in Z") -> 'unstructured'.
  - The user expressing preferences ("I prefer short answers",
    "I love refund data") -> 'unstructured'.
  - The user asking what the agent remembers / knows about them
    ("What do you remember about me?", "Do you know who I am?",
    "Remind me what you know") -> 'unstructured'.
- Only classify as out_of_scope if the question is BOTH unrelated to the
  dataset AND not about the user themselves.

Query-recommendation requests (also in-scope):
- When the user asks the agent for IDEAS about what to explore next,
  this is in-scope (the agent will look at history + profile and
  suggest a dataset query). Classify as 'structured'.
  Examples: "what should I query next?", "any suggestions?", "what
  else can I look at?", "got any ideas?", "where should I go from
  here?".
- Confirmations/refinements that come right after such a suggestion
  ("yes, do it", "go ahead", "I'd rather see examples", "what about
  SHIPPING?") are also 'structured' — they continue the same
  recommendation flow.

Return your classification and a one-sentence reason."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


# Build the router LLM once (module-level) so we don't pay the construction
# cost on every call.
_router_llm = get_llm(ROUTER_MODEL).with_structured_output(RouteDecision)


def _format_history(history: List[BaseMessage], max_turns: int = 4) -> str:
    """Format the last few messages as a short conversation transcript.

    Only includes the textual content of HumanMessage and AIMessage; skips
    tool calls and tool results to keep the router's input compact.
    """
    relevant: list[str] = []
    for msg in history:
        if isinstance(msg, HumanMessage) and msg.content:
            relevant.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage) and msg.content:
            # Strip any chain-of-thought tags some models emit.
            content = str(msg.content)
            if "</think>" in content:
                content = content.split("</think>", 1)[-1].strip()
            if content:
                relevant.append(f"Assistant: {content}")
    return "\n".join(relevant[-(2 * max_turns):])


def route_query(
    question: str,
    history: Optional[List[BaseMessage]] = None,
) -> RouteDecision:
    """Classify a user query into structured / unstructured / out_of_scope.

    Args:
        question: The user's latest question, in natural language.
        history: Optional prior messages from the same session. When the
            current question is a follow-up ("3 more", "what about X",
            "total of the last two"), this lets the router classify it
            correctly instead of treating it as out_of_scope.

    Returns:
        A RouteDecision with the category and a one-sentence reason.
    """
    user_content = question
    if history:
        transcript = _format_history(history)
        if transcript:
            user_content = (
                "Recent conversation (for context only):\n"
                f"{transcript}\n\n"
                f"Latest user question to classify:\n{question}"
            )
    messages = [
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]
    return _router_llm.invoke(messages)
