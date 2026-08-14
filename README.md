# QA Automation Agent — Week 1

An agent that takes a plain-English testing goal and executes it in a real
browser via Playwright, using a LangGraph ReAct loop instead of hardcoded
selectors.

## What this week's build does

Given a goal like *"search for 'locators' and confirm results appear"*, the
agent:
1. Looks at the live page (`get_page_elements`) instead of being told selectors
2. Decides what to click/fill based on visible text, placeholders, and names
3. Re-scans the page after every action (since ids can shift after navigation)
4. Verifies the outcome and reports PASS/FAIL with a plain-English reason

No selector is ever hardcoded or written by a human — the LLM reasons over
what's actually on the page. This is the foundation week 3's self-healing
logic builds on.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium     # downloads the browser binary
cp .env.example .env            # then add your OPENROUTER_API_KEY
```

## Run

```bash
python main.py
```

Or with your own goal:

```bash
python main.py "https://your-app.com" "Log in with username 'demo' and password 'demo123', then confirm the dashboard loads"
```

Set `HEADLESS=false` in `.env` for week 1 so you can watch the agent work —
seeing it reason through a real page is the best way to catch bad prompting
or flaky element detection early.

## Project structure

```
qa-automation-agent/
├── main.py                    # entry point
├── src/
│   ├── llm/gateway.py         # OpenRouter/DeepSeek wrapper (swap models here)
│   ├── tools/browser_tools.py # Playwright wrapped as LangChain tools
│   └── agent/graph.py         # LangGraph ReAct loop (agent <-> tools)
├── requirements.txt
└── .env.example
```

## Design notes worth remembering for interviews

- **Why data-qa-agent-id tagging instead of asking the LLM for CSS/XPath**:
  LLMs are unreliable at writing valid selectors, and hand-written selectors
  are exactly the brittleness this project exists to fix. Tagging visible
  interactive elements with a stable numeric id at scan-time turns "find the
  right element" into a much easier classification problem, and gives week 3's
  self-healing logic a consistent interface to build on.
- **Why temperature=0.0**: tool-calling agents need repeatable decisions, not
  creative variation — the same page state should produce the same action.
- **Why re-scan after every action**: SPA navigation and DOM mutations
  invalidate old element ids; treating the page as stale after any mutation
  is safer than trying to guess when a re-scan is needed.

## Next steps (weeks 2–4)

- **Week 2**: split the single ReAct node into explicit planner / executor /
  verifier nodes so failures can be attributed to the right stage
- **Week 3**: self-healing — when `click_element`/`fill_element` fails, catch
  it, re-scan, and have the LLM re-locate the element by its original
  description rather than immediately failing the test
- **Week 4**: eval harness (self-heal success rate, false positive/negative
  rate) + a short demo video showing a selector break → agent heals → test
  still passes
