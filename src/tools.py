"""Tools exposed to the ReAct agent for querying the Bitext dataset.

Each tool follows the same pattern:
    - A Pydantic BaseModel describing the input arguments (with field-level
      descriptions the LLM can read).
    - A @tool-decorated function with a detailed docstring that tells the LLM
      WHEN to use the tool, what it returns, and shows realistic examples.

Design principle (per the assignment): "A few well-designed tools beat many
poorly described ones." We keep the toolset small and composable so the agent
can chain them for multi-step reasoning.
"""

from typing import Any, Dict, List, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

from src.data_loader import load_dataset


# LLMs sometimes pass the literal strings 'null' / 'None' / '' instead of
# omitting an optional argument. Treat those as "no value".
_NULL_STRINGS = {"null", "none", "nan", "n/a", ""}


def _coerce_null_string(value: Any) -> Any:
    if isinstance(value, str) and value.strip().lower() in _NULL_STRINGS:
        return None
    return value


def _normalize_category(category: Optional[str]) -> Optional[str]:
    """Categories in the dataset are uppercase (e.g. REFUND, ACCOUNT).
    Normalize user input so the agent doesn't have to worry about casing.
    """
    return category.strip().upper() if category else None


def _normalize_intent(intent: Optional[str]) -> Optional[str]:
    """Intents in the dataset are lowercase with underscores (e.g. cancel_order)."""
    return intent.strip().lower() if intent else None


# ---------------------------------------------------------------------------
# Tool 1: list_categories
# ---------------------------------------------------------------------------


class ListCategoriesInput(BaseModel):
    """No arguments needed."""


@tool(args_schema=ListCategoriesInput)
def list_categories() -> List[str]:
    """Return the list of all high-level categories present in the dataset.

    Use this tool when the user asks what categories exist, what topics the
    dataset covers, or to discover valid category names before filtering.

    Example user questions this answers:
        - "What categories exist in the dataset?"
        - "What topics does the data cover?"

    Returns:
        A sorted list of unique category names (uppercase strings),
        e.g. ['ACCOUNT', 'CANCEL', 'DELIVERY', ...].
    """
    df = load_dataset()
    return sorted(df["category"].unique().tolist())


# ---------------------------------------------------------------------------
# Tool 2: list_intents
# ---------------------------------------------------------------------------


class ListIntentsInput(BaseModel):
    category: Optional[str] = Field(
        default=None,
        description=(
            "Optional. If provided, only intents belonging to this category are "
            "returned. Category is case-insensitive (e.g. 'ACCOUNT' or 'account'). "
            "OMIT this argument (do not pass 'null' or empty string) if you want all intents."
        ),
    )

    _coerce_category = field_validator("category", mode="before")(_coerce_null_string)


@tool(args_schema=ListIntentsInput)
def list_intents(category: Optional[str] = None) -> List[str]:
    """Return the list of intents (specific user goals) in the dataset.

    Use this tool when the user asks what intents exist, or to discover the
    exact intent name to use as a filter for other tools (count_rows,
    get_examples, etc.). If the user uses informal phrasing like "people
    wanting their money back", call this first (with the relevant category
    if you can guess it) to find the matching intent name like 'get_refund'.

    Example user questions this answers:
        - "What intents exist in the dataset?"
        - "What intents are in the ACCOUNT category?"
        - (Internal use) Finding the right intent name before filtering.

    Args:
        category: Optional category to filter by (case-insensitive).

    Returns:
        A sorted list of unique intent names (lowercase with underscores),
        e.g. ['cancel_order', 'change_order', ...].
    """
    df = load_dataset()
    cat = _normalize_category(category)
    if cat:
        df = df[df["category"] == cat]
    return sorted(df["intent"].unique().tolist())


# ---------------------------------------------------------------------------
# Tool 3: count_rows
# ---------------------------------------------------------------------------


class CountRowsInput(BaseModel):
    category: Optional[str] = Field(
        default=None,
        description=(
            "Optional. Filter to this category before counting (case-insensitive). "
            "E.g. 'REFUND', 'ACCOUNT'. OMIT this argument (do not pass 'null') "
            "for no category filter."
        ),
    )
    intent: Optional[str] = Field(
        default=None,
        description=(
            "Optional. Filter to this intent before counting (case-insensitive). "
            "E.g. 'cancel_order', 'get_refund'. OMIT this argument (do not pass "
            "'null') for no intent filter. Combine with category for more specific counts."
        ),
    )

    _coerce_category = field_validator("category", mode="before")(_coerce_null_string)
    _coerce_intent = field_validator("intent", mode="before")(_coerce_null_string)


@tool(args_schema=CountRowsInput)
def count_rows(
    category: Optional[str] = None, intent: Optional[str] = None
) -> Dict[str, object]:
    """Count rows in the dataset, optionally filtered by category and/or intent.

    Use this tool for any "how many" question. You can chain it after
    list_intents to first discover the right intent name, e.g.:
        list_intents(category='REFUND') -> ['check_refund_policy', 'get_refund', ...]
        count_rows(intent='get_refund') -> {'count': 1000, ...}

    Example user questions this answers:
        - "How many refund requests did we get?" -> count_rows(intent='get_refund')
        - "How many entries in ACCOUNT?" -> count_rows(category='ACCOUNT')
        - "How many complaints in FEEDBACK?" -> count_rows(category='FEEDBACK', intent='complaint')
        - "How many total rows?" -> count_rows()

    Args:
        category: Optional category filter.
        intent: Optional intent filter.

    Returns:
        A dict with keys:
            - 'count': the number of matching rows (int).
            - 'filters_applied': dict echoing the filters used, for clarity.
    """
    df = load_dataset()
    cat = _normalize_category(category)
    intnt = _normalize_intent(intent)

    if cat:
        df = df[df["category"] == cat]
    if intnt:
        df = df[df["intent"] == intnt]

    return {
        "count": int(len(df)),
        "filters_applied": {"category": cat, "intent": intnt},
    }


# ---------------------------------------------------------------------------
# Tool 4: get_examples
# ---------------------------------------------------------------------------


class GetExamplesInput(BaseModel):
    n: int = Field(
        default=3,
        description="Number of examples to return. Must be a positive integer. Typical values: 3-10.",
        ge=1,
        le=50,
    )
    category: Optional[str] = Field(
        default=None,
        description=(
            "Optional category filter (case-insensitive), e.g. 'SHIPPING'. "
            "OMIT this argument (do not pass 'null') for no category filter."
        ),
    )
    intent: Optional[str] = Field(
        default=None,
        description=(
            "Optional intent filter (case-insensitive), e.g. 'cancel_order'. "
            "OMIT this argument (do not pass 'null') for no intent filter. "
            "Combine with category for more targeted examples."
        ),
    )

    _coerce_category = field_validator("category", mode="before")(_coerce_null_string)
    _coerce_intent = field_validator("intent", mode="before")(_coerce_null_string)


@tool(args_schema=GetExamplesInput)
def get_examples(
    n: int = 3,
    category: Optional[str] = None,
    intent: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Return n example rows from the dataset, optionally filtered.

    Each example includes both the customer instruction and the agent
    response, so the user sees a full interaction. Use this for any
    "show me examples", "give me samples", or "what do they look like"
    request.

    Call this tool ONCE per user request. The returned rows ARE the
    examples — present them in your final answer, do not request more.

    Example user questions this answers:
        - "Show me 3 examples from the SHIPPING category."
            -> get_examples(n=3, category='SHIPPING')
        - "Give me 5 sample cancellation requests."
            -> get_examples(n=5, intent='cancel_order')
        - "Show me what refund-related queries look like."
            -> get_examples(n=5, category='REFUND')

    Args:
        n: How many examples to return (1-50).
        category: Optional category filter.
        intent: Optional intent filter.

    Returns:
        A list of dicts, each with keys: 'category', 'intent',
        'instruction', 'response'. Empty list if no rows match.
    """
    df = load_dataset()
    cat = _normalize_category(category)
    intnt = _normalize_intent(intent)

    if cat:
        df = df[df["category"] == cat]
    if intnt:
        df = df[df["intent"] == intnt]

    if df.empty:
        return []

    sample_size = min(n, len(df))
    sample = df.sample(n=sample_size)

    def _truncate(text: str, limit: int = 300) -> str:
        text = str(text)
        return text if len(text) <= limit else text[:limit].rstrip() + "..."

    return [
        {
            "category": str(row["category"]),
            "intent": str(row["intent"]),
            "instruction": _truncate(row["instruction"], 200),
            "response": _truncate(row["response"], 300),
        }
        for _, row in sample.iterrows()
    ]


# ---------------------------------------------------------------------------
# Tool 5: intent_distribution
# ---------------------------------------------------------------------------


class IntentDistributionInput(BaseModel):
    category: str = Field(
        description=(
            "The category to compute the intent distribution for "
            "(case-insensitive), e.g. 'ACCOUNT'."
        ),
    )


@tool(args_schema=IntentDistributionInput)
def intent_distribution(category: str) -> Dict[str, int]:
    """Return the distribution (intent -> row count) for a given category.

    Use this when the user asks about how queries are distributed within a
    category, what the breakdown looks like, or which intents are most
    common in a category.

    Example user questions this answers:
        - "What is the distribution of intents in the ACCOUNT category?"
            -> intent_distribution(category='ACCOUNT')
        - "What's the breakdown of REFUND queries?"
        - "Which intent is most common in PAYMENT?"

    Args:
        category: Category name (case-insensitive).

    Returns:
        A dict mapping intent name to row count, sorted by count
        descending. Empty dict if the category doesn't exist.
        Example: {'edit_account': 1100, 'create_account': 1000, ...}
    """
    df = load_dataset()
    cat = _normalize_category(category)
    df = df[df["category"] == cat]
    if df.empty:
        return {}
    counts = df["intent"].value_counts()
    return {intent: int(count) for intent, count in counts.items()}


# ---------------------------------------------------------------------------
# Tool 6: get_texts_for_summary
# ---------------------------------------------------------------------------


class GetTextsForSummaryInput(BaseModel):
    category: Optional[str] = Field(
        default=None,
        description=(
            "Optional category filter (case-insensitive). At least one of "
            "category or intent SHOULD be provided. OMIT this argument "
            "(do not pass 'null') for no category filter."
        ),
    )
    intent: Optional[str] = Field(
        default=None,
        description=(
            "Optional intent filter (case-insensitive). At least one of "
            "category or intent SHOULD be provided. OMIT this argument "
            "(do not pass 'null') for no intent filter."
        ),
    )
    n: int = Field(
        default=30,
        description=(
            "Number of rows to fetch for summarization. Default 30 is "
            "usually enough; raise to 50 for richer summaries."
        ),
        ge=5,
        le=100,
    )

    _coerce_category = field_validator("category", mode="before")(_coerce_null_string)
    _coerce_intent = field_validator("intent", mode="before")(_coerce_null_string)


@tool(args_schema=GetTextsForSummaryInput)
def get_texts_for_summary(
    category: Optional[str] = None,
    intent: Optional[str] = None,
    n: int = 30,
) -> List[Dict[str, str]]:
    """Fetch a batch of rows specifically for the LLM to read and summarize.

    Use this tool for OPEN-ENDED questions that require synthesizing or
    summarizing text content from the dataset, NOT for counting or listing.
    After calling this tool, YOU (the LLM) should read the returned texts
    and produce a natural-language summary in your final answer.

    Example user questions this answers:
        - "Summarize the FEEDBACK category."
            -> get_texts_for_summary(category='FEEDBACK', n=40)
        - "How do agents typically respond to cancellation requests?"
            -> get_texts_for_summary(intent='cancel_order', n=30)
        - "What's the tone of customer complaints?"
            -> get_texts_for_summary(intent='complaint', n=30)

    Args:
        category: Optional category filter.
        intent: Optional intent filter.
        n: Number of rows to sample (5-100, default 30).

    Returns:
        A list of dicts, each with 'instruction' and 'response' fields,
        ready for the LLM to read and summarize. Empty list if filters
        match no rows.
    """
    df = load_dataset()
    cat = _normalize_category(category)
    intnt = _normalize_intent(intent)

    if cat:
        df = df[df["category"] == cat]
    if intnt:
        df = df[df["intent"] == intnt]

    if df.empty:
        return []

    sample_size = min(n, len(df))
    sample = df.sample(n=sample_size)

    return [
        {
            "instruction": str(row["instruction"]),
            "response": str(row["response"]),
        }
        for _, row in sample.iterrows()
    ]


# ---------------------------------------------------------------------------
# Tool registry - import this list anywhere we need to bind tools to the agent
# ---------------------------------------------------------------------------

ALL_TOOLS = [
    list_categories,
    list_intents,
    count_rows,
    get_examples,
    intent_distribution,
    get_texts_for_summary,
]
