"""Query router: classifies an incoming user query into one of three buckets.

The router is the first node in the agent graph. It uses the small, fast
ROUTER_MODEL to decide whether the question is:
    - structured: can be answered with concrete data operations (count, filter, list)
    - unstructured: requires reading text and summarizing
    - out_of_scope: unrelated to the dataset, should be politely declined

Out-of-scope queries must be detected here so the agent never answers them
from general LLM knowledge.
"""

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
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

Return your classification and a one-sentence reason."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


# Build the router LLM once (module-level) so we don't pay the construction
# cost on every call.
_router_llm = get_llm(ROUTER_MODEL).with_structured_output(RouteDecision)


def route_query(question: str) -> RouteDecision:
    """Classify a user query into structured / unstructured / out_of_scope.

    Args:
        question: The user's question, in natural language.

    Returns:
        A RouteDecision with the category and a one-sentence reason.
    """
    messages = [
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=question),
    ]
    return _router_llm.invoke(messages)
