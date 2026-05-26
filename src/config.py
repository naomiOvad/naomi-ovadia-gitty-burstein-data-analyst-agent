"""Project configuration: model names, endpoints, and LLM factory.

All "magic strings" live here. Change a model name in one place and the
rest of the app picks it up.
"""

import os
from typing import Optional

import truststore
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Use the OS trust store for SSL verification (handles corporate proxies
# that inject self-signed certs). Must run before any HTTPS connections.
truststore.inject_into_ssl()

load_dotenv()


# ---------------------------------------------------------------------------
# Models — see README for the rationale of the two-model strategy.
# ---------------------------------------------------------------------------

# Small, fast model for the router node.
# Classification (structured / unstructured / out-of-scope) is a simple task,
# so we use Qwen3-30B-A3B — a Mixture-of-Experts model with only 3B active
# parameters, giving very low latency while staying smart enough for routing.
ROUTER_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"

# Larger model for the ReAct agent itself.
# Multi-step reasoning, tool selection, and summarization benefit from a
# strong instruction-following model. We use Qwen3-32B because in practice it
# follows the STOPPING RULES in the system prompt more reliably than Llama
# 3.3 70B, which tended to loop on get_examples when filtered by category.
AGENT_MODEL = "Qwen/Qwen3-32B"


# ---------------------------------------------------------------------------
# Agent loop limits
# ---------------------------------------------------------------------------

# Maximum number of ReAct iterations. Each iteration is one Think + Act cycle.
# Beyond this the agent returns a graceful fallback message.
MAX_ITERATIONS = 12


# ---------------------------------------------------------------------------
# Nebius Token Factory endpoint (OpenAI-compatible)
# ---------------------------------------------------------------------------

NEBIUS_BASE_URL = "https://api.studio.nebius.com/v1/"


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------


def get_llm(
    model: str,
    temperature: float = 0.0,
    api_key: Optional[str] = None,
) -> ChatOpenAI:
    """Create a ChatOpenAI client pointed at the Nebius Token Factory.

    Nebius exposes an OpenAI-compatible API, so we can reuse the standard
    langchain_openai.ChatOpenAI class — only the base_url and api_key change.

    Args:
        model: One of the Nebius model IDs (ROUTER_MODEL or AGENT_MODEL).
        temperature: Sampling temperature. Default 0.0 for deterministic
            tool-using behavior; raise for more creative summaries.
        api_key: Optional explicit API key. If not provided, reads
            NEBIUS_API_KEY from the environment (loaded from .env).

    Returns:
        A configured ChatOpenAI instance.

    Raises:
        ValueError: If no API key is available.
    """
    key = api_key or os.getenv("NEBIUS_API_KEY")
    if not key:
        raise ValueError(
            "NEBIUS_API_KEY is not set. Add it to your .env file: "
            "NEBIUS_API_KEY=your_key_here"
        )

    return ChatOpenAI(
        model=model,
        base_url=NEBIUS_BASE_URL,
        api_key=key,
        temperature=temperature,
    )
