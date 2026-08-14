# QA Automation Agent

An autonomous browser-testing agent that takes a natural-language QA goal (e.g. *"Log in with username 'demo' and password 'demo123', then confirm the dashboard loads"*) and executes it against a real browser — planning the steps, carrying them out, and independently verifying each one.

## How it works

The agent is built as a **LangGraph** state machine with three explicit stages, instead of one node that plans/acts/judges all at once. This keeps failures attributable — you can tell whether a run failed because of a bad plan, a bad action, or a bad verification, rather than just seeing "FAILED".

```
planner -> executor -> tools -> executor -> ... -> verifier
                                                  |
                              ┌───────────────────┼───────────────────┐
                              ▼                    ▼                   ▼
                          next_step            retry_step         finalize (END)
```

- **Planner** — turns the goal into a short ordered list (3–6) of concrete, verifiable UI steps. The last step is always a verification/confirmation check.
- **Executor** — carries out **one step at a time** using the browser tools (`get_page_elements`, `click_element`, `fill_element`, `press_key`, `assert_text_present`, `get_current_url`, `navigate`).
- **Verifier** — independently judges PASS/FAIL for that step by reading the actual tool-call transcript, not the executor's self-reported summary. On FAIL, the step is retried (up to `MAX_STEP_ATTEMPTS`, default 2) before the whole run is marked FAILED.

Retries are **per-step**, not per-run — if step 3 of 5 fails twice, only step 3 is retried; steps 1–2 aren't re-run.

## Self-healing element location

Rather than asking the LLM to write raw CSS/XPath selectors (brittle, and the exact problem this project exists to solve), every interactive element on the page is tagged with a stable `data-qa-agent-id` attribute at scan time. The agent refers to elements only by that id.

IDs are only stable *within a single scan*. If the page re-renders between scanning and acting (SPA re-render, modal opening, ad loading, etc.), the id the agent picked may go stale. When that happens:

1. The failed action's exception is caught instead of crashing the run.
2. The page is re-scanned.
3. An LLM is asked to re-locate the **same element** using the human-readable description the agent originally gave for it (its text/placeholder/name) — not the id, since the id is exactly what went stale.
4. If a confident match is found, the action is retried against the new id.

Every heal attempt (successful or not) is recorded in `HEAL_LOG` for later reporting/eval via `get_heal_log()`.

## Why async Playwright

Playwright's **sync** API binds its browser driver to whichever thread started it. LangGraph's tool-execution node can dispatch tool calls on a different worker thread than the one that launched the browser, which corrupts the sync driver. Using the **async** API and running the whole agent on a single asyncio event loop (`graph.ainvoke`) avoids crossing threads entirely.

## Project structure

```
main.py                    # Entry point / CLI runner
src/
├── agent/
│   └── graph.py            # LangGraph state machine: planner / executor / verifier
├── tools/
│   └── browser_tools.py    # Playwright-backed LangChain tools + self-healing logic
└── llm/
    └── gateway.py           # LLM client provider (get_llm) — used by planner, executor, verifier, and the self-heal matcher
```

## Requirements

- Python 3.10+
- [`langchain-core`](https://pypi.org/project/langchain-core/)
- [`langgraph`](https://pypi.org/project/langgraph/)
- [`playwright`](https://pypi.org/project/playwright/) (with browsers installed: `playwright install`)
- An LLM provider configured behind `src/llm/gateway.get_llm()`

## Setup

```bash
pip install langchain-core langgraph playwright
playwright install chromium
```

Configure whatever credentials/model your `src/llm/gateway.py` implementation needs (e.g. API keys as environment variables).

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `HEADLESS` | `false` | Set to `true` to run the browser headlessly. |

## Usage

Run with the default target (playwright.dev) and default goal:

```bash
python main.py
```

Run against your own app with a custom goal:

```bash
python main.py "https://your-app.com" "Log in with username 'demo' and password 'demo123', then confirm the dashboard loads"
```

The agent will print a step-by-step log and an overall `PASS`/`FAIL` result, e.g.:

```
--- Running agent ---
Start URL: https://your-app.com
Goal: Log in with username 'demo' and password 'demo123', then confirm the dashboard loads

--- Agent finished ---

Goal: Log in with username 'demo' and password 'demo123', then confirm the dashboard loads
Overall result: PASS

Step-by-step log:
  [PASS] Step 1 (attempt 1): Find the username and password fields and enter the credentials -- Fields were filled with the given values.
  [PASS] Step 2 (attempt 1): Submit the login form -- Form was submitted via Enter key press.
  [PASS] Step 3 (attempt 1): Confirm the dashboard loads -- The word 'Dashboard' was found on the page.
```

## Available tools

| Tool | Purpose |
|---|---|
| `navigate(url)` | Navigate the browser to a URL |
| `get_page_elements()` | Scan the page and return every visible interactive element with a stable id |
| `click_element(element_id, element_description)` | Click an element by id; self-heals if the id is stale |
| `fill_element(element_id, text, element_description)` | Fill an input/textarea by id; self-heals if the id is stale |
| `press_key(element_id, key, element_description)` | Press a key (e.g. `Enter`) on a focused element; self-heals if the id is stale |
| `assert_text_present(expected_text)` | Check whether text appears anywhere on the current page |
| `get_current_url()` | Return the current page URL |

## Notes

- The verifier is intentionally strict: if the transcript doesn't clearly show the expected outcome, it treats the step as `FAIL` rather than trusting the executor's own claim of success.
- `MAX_STEP_ATTEMPTS` (in `src/agent/graph.py`) controls how many times a single failed step is retried before the whole run is marked `FAILED`.
