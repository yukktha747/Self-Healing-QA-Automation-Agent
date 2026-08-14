"""
Agent Graph — Planner / Executor / Verifier
---------------------------------------------------

A single node that plans, acts, and judges success all at once makes
failures hard to attribute: if a run ends in FAILED, you don't know whether
the agent misunderstood the goal, clicked the wrong element, or the
verification logic itself was too strict.

This graph splits that into three explicit stages:

    planner   -- turns the goal into an ordered list of concrete steps
    executor  -- carries out ONE step at a time using the browser tools
    verifier  -- independently checks whether that step actually succeeded

                 ┌─────────────────────────────┐
                 ▼                             │ (tool_calls present)
    planner -> executor -> tools ──────────────┘
                 │
                 │ (no more tool_calls -> step attempt finished)
                 ▼
             verifier
                 │
      ┌──────────┼──────────────┐
      ▼          ▼              ▼
   next step   retry step    FAIL / all steps
   (executor)  (executor)    passed -> END

Retries are per-step (capped by MAX_STEP_ATTEMPTS), not whole-run retries --
if step 3 of 5 fails twice, we don't re-run steps 1-2, we fail fast with a
clear "step 3 failed after 2 attempts" reason.
"""

import json
import re
from typing import Annotated, Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from src.llm.gateway import get_llm
from src.tools.browser_tools import ALL_TOOLS

MAX_STEP_ATTEMPTS = 2

PLANNER_SYSTEM_PROMPT = """You are a QA test planner. Given a testing goal, break it \
into a short ordered list of concrete, verifiable UI steps.

Rules:
- Each step should be one user action or one check, not both.
- The LAST step must always be a verification/confirmation step describing \
what should be true if the goal succeeded.
- Keep it to 3-6 steps. Do not pad with unnecessary steps.
- Respond with ONLY a JSON array of strings. No prose, no markdown fences.

Example goal: "Search for 'locators' and confirm results appear"
Example response:
["Find the search input on the page", "Type 'locators' into the search input and submit it", "Confirm the word 'Locators' appears in the results"]
"""

EXECUTOR_SYSTEM_PROMPT = """You are a QA automation agent executing ONE step of a \
larger test plan. You control a real browser through tools.

Always follow this pattern:
1. Call get_page_elements to see what's currently on the page before acting.
2. Pick the correct element by its id and text/placeholder/name, then \
click_element or fill_element as needed.
3. If a step requires SUBMITTING a search box or form and there's no separate \
visible submit/search button to click, use press_key with key="Enter" on the \
input you just filled -- typing text alone does not submit anything.
4. ALWAYS pass element_description to click_element/fill_element/press_key -- a \
short phrase using the element's actual text/placeholder/name from \
get_page_elements (e.g. "Search button" or "placeholder='Email address'"). This is \
used to automatically re-locate the element if the page changes between your scan \
and your action, so be specific and use wording that will still make sense if the \
id changes.
5. After an action that changes the page, call get_page_elements again -- ids \
can change after navigation.
6. If a tool result says "Self-healed", the action still succeeded despite the \
element moving -- treat it as a normal success and continue.

Only do what THIS step requires -- do not attempt later steps. Once you've \
carried out the step's action, stop calling tools and reply with a brief plain-text \
description of what you did. Do not judge PASS/FAIL yourself -- that is decided \
separately.
"""

VERIFIER_SYSTEM_PROMPT = """You are a strict QA verifier. You will be shown one step \
from a test plan and a transcript of what the executor did (including tool call \
results) while attempting it.

Decide whether the step's goal was actually achieved, based only on the evidence in \
the transcript -- not on what the executor claims it did. If the transcript doesn't \
clearly show the expected outcome, treat it as FAIL.

Respond in EXACTLY this format, nothing else:
STATUS: PASS
REASON: <one short sentence>

or

STATUS: FAIL
REASON: <one short sentence explaining what evidence was missing or wrong>
"""


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    goal: str
    plan: list[str]
    step_index: int
    step_attempts: int
    step_log: list[dict]
    final_status: Optional[str]
    final_summary: Optional[str]


def _parse_plan(raw: str) -> list[str]:
    """Parse the planner's JSON array response, tolerating stray markdown fences."""
    cleaned = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        plan = json.loads(cleaned)
        if isinstance(plan, list) and all(isinstance(s, str) for s in plan):
            return plan
    except json.JSONDecodeError:
        pass
    # Fallback: treat non-empty lines as steps if JSON parsing fails.
    lines = [line.strip("-*0123456789. ").strip() for line in cleaned.splitlines()]
    return [line for line in lines if line]


async def planner_node(state: AgentState) -> AgentState:
    llm = get_llm()
    response = await llm.ainvoke(
        [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=f"Goal: {state['goal']}"),
        ]
    )
    plan = _parse_plan(response.content)

    step_intro = HumanMessage(
        content=f"Step 1 of {len(plan)}: {plan[0]}\n\nExecute this step using the tools."
    )

    return {
        "messages": [SystemMessage(content=EXECUTOR_SYSTEM_PROMPT), step_intro],
        "plan": plan,
        "step_index": 0,
        "step_attempts": 1,
        "step_log": [],
        "final_status": None,
        "final_summary": None,
    }


async def executor_node(state: AgentState) -> AgentState:
    llm = get_llm().bind_tools(ALL_TOOLS)
    response = await llm.ainvoke(state["messages"])
    return {"messages": [response]}


def should_continue_executing(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return "verifier"


async def verifier_node(state: AgentState) -> AgentState:
    step = state["plan"][state["step_index"]]

    # Give the verifier the step under test plus everything since the plan
    # was made -- this includes every tool call and tool result, which is
    # the actual evidence, not just the executor's self-report.
    transcript = state["messages"]

    llm = get_llm()
    response = await llm.ainvoke(
        [
            SystemMessage(content=VERIFIER_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Step under test: {step}\n\n"
                    f"Transcript:\n"
                    + "\n".join(
                        f"[{m.type}] {m.content}"
                        for m in transcript
                        if getattr(m, "content", None)
                    )
                )
            ),
        ]
    )

    status_match = re.search(r"STATUS:\s*(PASS|FAIL)", response.content, re.IGNORECASE)
    reason_match = re.search(r"REASON:\s*(.+)", response.content)
    status = status_match.group(1).upper() if status_match else "FAIL"
    reason = reason_match.group(1).strip() if reason_match else "Verifier gave an unparseable response."

    log_entry = {
        "step_index": state["step_index"],
        "step": step,
        "attempt": state["step_attempts"],
        "status": status,
        "reason": reason,
    }

    return {"messages": [], "step_log": state["step_log"] + [log_entry]}


def route_after_verifier(state: AgentState) -> str:
    last_log = state["step_log"][-1]
    is_last_step = state["step_index"] == len(state["plan"]) - 1

    if last_log["status"] == "PASS":
        return "END" if is_last_step else "next_step"

    # FAIL
    if state["step_attempts"] < MAX_STEP_ATTEMPTS:
        return "retry_step"
    return "END"


def prepare_next_step(state: AgentState) -> AgentState:
    next_index = state["step_index"] + 1
    next_step = state["plan"][next_index]
    intro = HumanMessage(
        content=f"Step {next_index + 1} of {len(state['plan'])}: {next_step}\n\nExecute this step using the tools."
    )
    return {"messages": [intro], "step_index": next_index, "step_attempts": 1}


def prepare_retry_step(state: AgentState) -> AgentState:
    last_log = state["step_log"][-1]
    step = state["plan"][state["step_index"]]
    intro = HumanMessage(
        content=(
            f"Your previous attempt at this step FAILED. Reason: {last_log['reason']}\n\n"
            f"Retry step {state['step_index'] + 1}: {step}\n\n"
            "Check the current page state again before acting -- it may have changed."
        )
    )
    return {"messages": [intro], "step_attempts": state["step_attempts"] + 1}


def finalize(state: AgentState) -> AgentState:
    # Overall PASS only if we actually reached and passed the final step.
    reached_last_step = state["step_index"] == len(state["plan"]) - 1
    last_step_passed = bool(state["step_log"]) and state["step_log"][-1]["status"] == "PASS"
    final_status = "PASS" if reached_last_step and last_step_passed else "FAIL"

    lines = [f"Goal: {state['goal']}", f"Overall result: {final_status}", "", "Step-by-step log:"]
    for entry in state["step_log"]:
        lines.append(
            f"  [{entry['status']}] Step {entry['step_index'] + 1} (attempt {entry['attempt']}): "
            f"{entry['step']} -- {entry['reason']}"
        )
    summary = "\n".join(lines)

    return {"final_status": final_status, "final_summary": summary}


def build_graph():
    tool_node = ToolNode(ALL_TOOLS)

    graph = StateGraph(AgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("tools", tool_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("next_step", prepare_next_step)
    graph.add_node("retry_step", prepare_retry_step)
    graph.add_node("finalize", finalize)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "executor")

    graph.add_conditional_edges(
        "executor", should_continue_executing, {"tools": "tools", "verifier": "verifier"}
    )
    graph.add_edge("tools", "executor")

    graph.add_conditional_edges(
        "verifier",
        route_after_verifier,
        {"next_step": "next_step", "retry_step": "retry_step", "END": "finalize"},
    )
    graph.add_edge("next_step", "executor")
    graph.add_edge("retry_step", "executor")
    graph.add_edge("finalize", END)

    return graph.compile()
