"""
QA Automation Agent -- entry point.

Run with:
    python main.py
    python main.py "https://your-app.com" "Log in with username 'demo' and password 'demo123', then confirm the dashboard loads"

The goal is first broken into an explicit step-by-step plan (planner), each
step is executed one at a time using the browser tools (executor), and each
attempt is independently verified against the actual tool-call transcript
(verifier) -- not the executor's self-report. Failed steps get up to
MAX_STEP_ATTEMPTS retries before the whole run is marked FAILED, with a
step-by-step log showing exactly which step failed and why.
"""

import asyncio
import sys
from langchain_core.messages import HumanMessage

from src.agent.graph import build_graph
from src.tools.browser_tools import get_session

DEFAULT_START_URL = "https://playwright.dev"
DEFAULT_GOAL = (
    "Use the search feature to search for 'locators', then confirm the "
    "word 'Locators' appears somewhere in the results."
)


async def run() -> None:
    start_url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_START_URL
    goal = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_GOAL

    session = get_session()
    await session.start()

    try:
        await session.page.goto(start_url, wait_until="domcontentloaded")

        graph = build_graph()

        print(f"\n--- Running agent ---\nStart URL: {start_url}\nGoal: {goal}\n")

        final_state = await graph.ainvoke(
            {
                "messages": [HumanMessage(content=f"Starting page: {start_url}")],
                "goal": goal,
            },
            config={"recursion_limit": 50},
        )

        print("--- Agent finished ---\n")
        print(final_state["final_summary"])

    finally:
        await session.stop()


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()