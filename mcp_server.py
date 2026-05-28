"""FastMCP server exposing the Bitext data-analyst tools.

This is a standalone process — separate from the LangGraph CLI agent in
main.py. It speaks the Model Context Protocol (over STDIO, the MCP
default) so any MCP-compatible client — Claude Desktop, Cursor, a
custom Python client using `fastmcp.Client` — can call the same data
tools the agent uses.

The tool implementations mirror those in src/tools.py but expose them
through the MCP protocol instead of LangChain's @tool decorator.

Run:
    python mcp_server.py        # STDIO transport
"""

from __future__ import annotations

from typing import Optional

from fastmcp import FastMCP

from src.data_loader import load_dataset
from src.tools import _normalize_category, _normalize_intent


mcp = FastMCP("Bitext Customer-Support Data Analyst")


# ---------------------------------------------------------------------------
# Tool 1: list_categories
# ---------------------------------------------------------------------------


@mcp.tool
def list_categories() -> list[str]:
    """Return the list of all high-level categories present in the dataset.

    Use this tool when the user asks what categories exist, what topics
    the data covers, or to discover valid category names before filtering.

    Returns:
        A sorted list of unique category names (uppercase strings),
        e.g. ['ACCOUNT', 'CANCEL', 'DELIVERY', ...].
    """
    df = load_dataset()
    return sorted(df["category"].unique().tolist())


# ---------------------------------------------------------------------------
# Tool 2: list_intents
# ---------------------------------------------------------------------------


@mcp.tool
def list_intents(category: Optional[str] = None) -> list[str]:
    """Return the list of intents (specific user goals) in the dataset.

    Use this tool when the user asks what intents exist, or to discover
    the exact intent name to use as a filter for other tools (count_rows,
    get_examples, etc.).

    Args:
        category: Optional category to filter by (case-insensitive).
            OMIT this argument for all intents.

    Returns:
        A sorted list of unique intent names (lowercase with underscores).
    """
    df = load_dataset()
    cat = _normalize_category(category)
    if cat:
        df = df[df["category"] == cat]
    return sorted(df["intent"].unique().tolist())


# ---------------------------------------------------------------------------
# Tool 3: count_rows
# ---------------------------------------------------------------------------


@mcp.tool
def count_rows(
    category: Optional[str] = None,
    intent: Optional[str] = None,
) -> dict:
    """Count rows in the dataset, optionally filtered by category and intent.

    Use this for any 'how many' question. Both filters are optional.

    Args:
        category: Optional category filter (case-insensitive).
        intent: Optional intent filter (case-insensitive).

    Returns:
        A dict with 'count' (int) and 'filters_applied' (dict echoing the
        normalized filters used).
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


@mcp.tool
def get_examples(
    n: int = 3,
    category: Optional[str] = None,
    intent: Optional[str] = None,
) -> list[dict]:
    """Return n example rows from the dataset, optionally filtered.

    Each example includes both the customer instruction and the agent
    response. Call this tool ONCE per request — the returned rows ARE
    the examples; don't request more.

    Args:
        n: How many examples to return (1-50).
        category: Optional category filter.
        intent: Optional intent filter.

    Returns:
        A list of dicts each with 'category', 'intent', 'instruction',
        'response'. Empty list if no rows match.
    """
    if n < 1:
        n = 1
    if n > 50:
        n = 50

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

    def _truncate(text: str, limit: int) -> str:
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


@mcp.tool
def intent_distribution(category: str) -> dict[str, int]:
    """Return the distribution (intent -> row count) for a given category.

    Use this when the user asks about how queries are distributed within a
    category, what the breakdown looks like, or which intents are most
    common in a category.

    Args:
        category: Category name (case-insensitive).

    Returns:
        A dict mapping intent name to row count, sorted by count
        descending. Empty dict if the category doesn't exist.
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


@mcp.tool
def get_texts_for_summary(
    category: Optional[str] = None,
    intent: Optional[str] = None,
    n: int = 30,
) -> list[dict]:
    """Fetch a batch of rows for the calling client to read and summarize.

    Use for OPEN-ENDED questions that require synthesizing text content
    (e.g. 'summarize FEEDBACK', 'how do agents respond to cancellations').

    Args:
        category: Optional category filter.
        intent: Optional intent filter.
        n: Number of rows to sample (5-100, default 30).

    Returns:
        A list of dicts each with 'instruction' and 'response'.
    """
    if n < 5:
        n = 5
    if n > 100:
        n = 100

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
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    mcp.run()  # STDIO transport — the MCP default
