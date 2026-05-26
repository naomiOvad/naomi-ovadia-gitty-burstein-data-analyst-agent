# Customer Service Data Analyst Agent

A LangGraph-based ReAct agent that answers questions about the
[Bitext Customer Support](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
dataset (26,872 customer/agent interactions across 11 categories and
27 intents).

The agent handles three kinds of queries:

- **Structured** — concrete questions answered by filtering/counting/sampling
  the data ("How many refund requests?", "Show me 5 SHIPPING examples").
- **Unstructured** — open-ended questions answered by reading and
  summarizing text content ("Summarize the FEEDBACK category").
- **Out-of-scope** — anything unrelated to the dataset is politely
  declined ("Who is the president of France?").

This is the Task 1 deliverable. Tasks 2 (memory) and 3 (MCP server)
will be added on top of this foundation.

---

## Setup (5 minutes)

### Prerequisites

- Python 3.10 or newer (tested on 3.12).
- A [Nebius Token Factory](https://tokenfactory.nebius.com) API key.

### Install

```bash
# 1. Clone or unzip the repo, then enter the directory
cd "naomi submission"

# 2. Create a virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate          # on Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

# 3. Set your Nebius API key
cp .env.example .env
# Open .env and replace the placeholder with your real key:
#   NEBIUS_API_KEY=your_real_key_here
```

### First run

```bash
python main.py
```

On the first run the agent will download the Bitext CSV (~20 MB) into
`data/bitext_dataset.csv`. Subsequent runs read from that cache.

You should land in an interactive prompt:

```
Customer Service Data Analyst Agent
Ask me anything about the Bitext customer-support dataset:
  - "What categories exist?"
  - "How many refund requests?"
  - "Summarize the FEEDBACK category."
Commands: /help, /exit, /quit

You: ▮
```

Type `/exit` (or Ctrl+D) to quit.

---

## How the agent works

### Graph

```
       [START]
          │
          ▼
      [ Router ]                 ← classifies the query
          │
   ┌──────┼──────┐
   ▼             ▼
[ Decline ]   [ ReAct Agent ]    ← out_of_scope vs. (un)structured
   │             │
   └──────┬──────┘
          ▼
        [END]
```

1. **Router node** classifies the incoming question as
   `structured` / `unstructured` / `out_of_scope` using a small,
   fast model with a Pydantic-enforced output schema.
2. If `out_of_scope`, the **Decline node** returns a polite refusal
   without consulting any tools or general knowledge.
3. Otherwise, the **ReAct agent** (LangGraph's `create_react_agent`)
   takes over: it picks a tool, observes the result, may chain to
   another tool, and produces a final natural-language answer.

A `recursion_limit` is set on the outer graph (`MAX_ITERATIONS * 2 + 4`).
If the agent doesn't reach a final answer in time, a graceful fallback
message is returned instead of an infinite loop.

### Models (Nebius Token Factory)

Two models are used, each suited to its role:

| Role | Model | Why |
|------|-------|-----|
| Router | `Qwen/Qwen3-30B-A3B-Instruct-2507` | Mixture-of-Experts with only 3B active parameters — very low latency, plenty smart for a 3-way classification. |
| Agent | `Qwen/Qwen3-32B` | Strong instruction-following and reliable tool-calling. In practice it follows the "stop after a successful tool call" rule more reliably than Llama 3.3 70B, which tended to loop on `get_examples` when filtering by category. |

Both are accessed via Nebius's OpenAI-compatible API
(`https://api.studio.nebius.com/v1/`).

### Tools

Six tools, all with Pydantic input schemas and detailed `WHEN-TO-USE`
docstrings (see [`src/tools.py`](src/tools.py)):

| Tool | Returns | When to use |
|------|---------|-------------|
| `list_categories` | list of category names | "What categories exist?" |
| `list_intents` | list of intent names, optionally per category | Discover the right intent name before filtering. |
| `count_rows` | `{count, filters_applied}` | "How many X?" questions. |
| `get_examples` | sample rows (instruction + response) | "Show me N examples of X". |
| `intent_distribution` | intent → count for one category | "What's the breakdown of X?". |
| `get_texts_for_summary` | a batch of rows for the LLM to summarize | Open-ended "summarize / how do agents respond" questions. |

Each tool's docstring also tells the model to OMIT optional arguments
(rather than passing the string `'null'`), and the schemas use
`field_validator`s that coerce `'null'`/`'none'`/`''` to `None` as a
safety net.

The toolset is intentionally small and composable. Per the assignment:
"A few well-designed tools beat many poorly described ones." The
example multi-step path the assignment hints at —
`list_intents('REFUND') → get_examples(intent='get_refund')` — works
out of the box (see Test 5 below).

### Reasoning trace in the CLI

The CLI streams the agent's reasoning to the terminal as it happens:

- `[Router] structured — reason` — the classification.
- `[Tool call] count_rows(intent='get_refund')` — each tool invocation.
- `[Result of count_rows] {...}` — the tool's output (truncated).
- `🤖 ...` — the final answer.

This satisfies the "print reasoning steps, not just the final answer"
requirement.

---

## Repo layout

```
naomi submission/
├── data/
│   └── bitext_dataset.csv      # downloaded on first run
├── src/
│   ├── __init__.py
│   ├── config.py               # model names + ChatOpenAI factory
│   ├── data_loader.py          # downloads / caches the CSV
│   ├── tools.py                # 6 tools + Pydantic input schemas
│   ├── router.py               # query classifier (structured/...)
│   ├── agent.py                # LangGraph wiring + run_agent()
│   └── cli.py                  # interactive REPL with reasoning trace
├── main.py                     # entry point (python main.py)
├── requirements.txt
├── .env.example
├── .gitignore
├── tests_output.txt            # captured run of all 8 example queries
├── PLAN.md                     # planning document (kept for reference)
└── README.md
```

---

## Example queries

These are the eight example queries from the assignment, each of
which the agent answers correctly. The full captured trace is in
[`tests_output.txt`](tests_output.txt).

| # | Query | Expected route | Notes |
|---|-------|---------------|-------|
| 1 | What categories exist in the dataset? | structured | One tool call (`list_categories`). |
| 2 | How many refund requests did we get? | structured | `count_rows(intent='get_refund')` → 997. |
| 3 | Show me 5 examples of the SHIPPING category. | structured | `get_examples(n=5, category='SHIPPING')`. |
| 4 | Summarize how agents respond to complaint intents. | unstructured | `get_texts_for_summary(intent='complaint', n=30)` then the LLM summarizes. |
| 5 | Show me examples of people wanting their money back. | structured | **Multi-step**: `list_intents('REFUND')` → `get_examples(intent='get_refund')`. |
| 6 | What is the distribution of intents in the ACCOUNT category? | structured | One call to `intent_distribution`. |
| 7 | What's the best CRM software for handling complaints? | out_of_scope | Declined politely, no tools called. |
| 8 | Who is the president of France? | out_of_scope | Declined politely, no tools called. |

To re-run them yourself:

```bash
python main.py
# then paste each question, one per line
```

### Testing the fallback

To see the max-iterations fallback message in action, temporarily
lower `MAX_ITERATIONS` in [`src/config.py`](src/config.py) to `2`
and re-run a complex query like "How many refund requests did we get?".
You should see:

```
🤖 I couldn't reach a final answer within the iteration limit
(2 steps). Could you rephrase your question or break it into
smaller parts?
```

---

## What's next (Tasks 2 & 3)

The CLI already accepts a `--session <id>` argument for forward
compatibility:

```bash
python main.py --session my_session
```

In Task 1 this has no effect, but Task 2 will hook a LangGraph
checkpointer (likely `SqliteSaver`) onto the graph and use the
session ID as the thread ID, giving the agent persistent
conversation memory across restarts.

Task 3 will expose three of the tools (`list_categories`,
`count_rows`, `get_examples`) over a FastMCP server, with a short
client snippet in this README's "How to connect" section.

---

## Troubleshooting

- **`AuthenticationError: 401`** — your `NEBIUS_API_KEY` is missing,
  expired, or for a different Nebius product. Generate a new one at
  [tokenfactory.nebius.com](https://tokenfactory.nebius.com).
- **`SSL: CERTIFICATE_VERIFY_FAILED`** on a corporate network —
  `truststore` (in requirements.txt) is included specifically to fix
  this; make sure `pip install -r requirements.txt` ran to completion.
- **Hugging Face download hangs** at 0% — handled. We download the CSV
  directly via `pandas.read_csv(URL)` instead of through the
  `datasets` library, which avoids HF Hub's rate limit on
  unauthenticated requests.
