"""
Browser Tools
-------------
Wraps Playwright (ASYNC API) as a set of LangChain tools an LLM agent can call.

Why async instead of sync: Playwright's sync API binds its internal browser
driver to whichever thread started it. LangGraph's tool-execution node can
dispatch tool calls on a different worker thread than the one that launched
the browser, which corrupts the sync driver ("cannot switch to a different
thread"). Using the async API and running the whole agent on a single asyncio
event loop (graph.ainvoke) avoids crossing threads entirely.

Design: instead of asking the LLM to write raw CSS/XPath selectors (which
it's bad at, and which breaks constantly -- the exact problem this project
exists to solve), we inject a `data-qa-agent-id` attribute onto every
interactive element on the page. The agent refers to elements by that
stable, numeric id.

Self-healing: ids are only stable *within one scan*. If the page re-renders
between "get_page_elements" and "click_element" (SPA re-render, a modal
opening, an ad loading, etc.), the id the agent picked can point at nothing,
or at the wrong element. Rather than letting that raw Playwright exception
kill the whole test step, click_element/fill_element catch the failure,
re-scan the page, and ask an LLM to re-locate the *same element* by the
description the agent originally gave for it (its text/placeholder/name) --
not by id, since the id is exactly what went stale. Every heal attempt is
logged to HEAL_LOG for later reporting/eval (see get_heal_log).
"""

import os
from typing import Optional
from langchain_core.tools import tool
from playwright.async_api import async_playwright, Page, Browser, Playwright, TimeoutError as PWTimeoutError


class BrowserSession:
    """Holds the single live Playwright/browser/page instance for a run."""

    def __init__(self) -> None:
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self.page: Optional[Page] = None

    async def start(self) -> None:
        headless = os.getenv("HEADLESS", "false").lower() == "true"
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=headless)
        self.page = await self._browser.new_page()

    async def stop(self) -> None:
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

    async def tag_interactive_elements(self) -> list[dict]:
        """
        Injects data-qa-agent-id on every interactive element and returns a
        simplified list describing each one, so the LLM can decide what to
        act on without ever seeing raw HTML/CSS.
        """
        assert self.page is not None, "Browser session not started"

        elements = await self.page.evaluate(
            """
            () => {
                const selector = 'a, button, input, textarea, select, [role="button"], [role="link"]';
                const nodes = Array.from(document.querySelectorAll(selector));
                return nodes.map((el, idx) => {
                    el.setAttribute('data-qa-agent-id', String(idx));
                    const rect = el.getBoundingClientRect();
                    return {
                        id: String(idx),
                        tag: el.tagName.toLowerCase(),
                        type: el.getAttribute('type') || null,
                        text: (el.innerText || el.value || '').trim().slice(0, 80),
                        placeholder: el.getAttribute('placeholder') || null,
                        name: el.getAttribute('name') || null,
                        visible: rect.width > 0 && rect.height > 0
                    };
                }).filter(e => e.visible);
            }
            """
        )
        return elements


# Single shared session used by all tool functions below.
_session = BrowserSession()

# Every self-heal attempt (successful or not) gets recorded here. This is
# the raw data the week 4 eval harness will summarize into a heal success
# rate; for now, get_heal_log() lets you inspect it after a run.
HEAL_LOG: list[dict] = []


def get_session() -> BrowserSession:
    return _session


def get_heal_log() -> list[dict]:
    return HEAL_LOG


def _format_elements_for_prompt(elements: list[dict]) -> str:
    lines = []
    for el in elements:
        desc = f"id={el['id']} <{el['tag']}"
        if el["type"]:
            desc += f" type={el['type']}"
        desc += ">"
        if el["text"]:
            desc += f' text="{el["text"]}"'
        if el["placeholder"]:
            desc += f' placeholder="{el["placeholder"]}"'
        if el["name"]:
            desc += f' name="{el["name"]}"'
        lines.append(desc)
    return "\n".join(lines)


async def _relocate_element(description: str, action: str) -> Optional[str]:
    """
    Re-scans the page and asks the LLM to find the element matching
    `description` (the text/placeholder/name the agent originally used to
    identify the element it wanted to act on). Returns the new element id,
    or None if no confident match is found.

    Imported lazily to avoid a circular import (llm.gateway doesn't depend
    on tools, but keeping the import local here makes that explicit).
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    from src.llm.gateway import get_llm

    elements = await _session.tag_interactive_elements()
    if not elements:
        return None

    prompt = (
        f"An automated test tried to {action} an element originally described as: "
        f"\"{description}\". That element is no longer at its expected location "
        f"(the page likely re-rendered). Here is the CURRENT list of visible "
        f"interactive elements on the page:\n\n{_format_elements_for_prompt(elements)}\n\n"
        f"Which id is the SAME element (or the closest reasonable match)? "
        f"Respond with ONLY the id number, or NONE if there is no reasonable match."
    )

    llm = get_llm()
    response = await llm.ainvoke(
        [
            SystemMessage(
                content="You are matching a stale element reference to its current location "
                "on a web page after a DOM change. Be precise -- if nothing clearly matches, say NONE."
            ),
            HumanMessage(content=prompt),
        ]
    )

    candidate = response.content.strip()
    valid_ids = {el["id"] for el in elements}
    if candidate in valid_ids:
        return candidate
    return None


@tool
async def navigate(url: str) -> str:
    """Navigate the browser to the given URL. Always use a full URL including https://."""
    page = _session.page
    await page.goto(url, wait_until="domcontentloaded")
    title = await page.title()
    return f"Navigated to {url}. Current title: {title}"


@tool
async def get_page_elements() -> str:
    """
    Scan the current page and return every visible interactive element
    (buttons, links, inputs, textareas, selects) along with a stable
    numeric id. Call this BEFORE clicking or filling anything, and again
    any time the page might have changed (e.g. after a click or navigation).
    """
    elements = await _session.tag_interactive_elements()
    if not elements:
        return "No interactive elements found on the current page."
    return _format_elements_for_prompt(elements)


@tool
async def click_element(element_id: str, element_description: str) -> str:
    """
    Click the interactive element with the given id (from get_page_elements).
    Always also pass element_description: a short phrase describing the
    element (its text, placeholder, or name as shown by get_page_elements) --
    e.g. "Search button" or "text='Sign in'". This is used to re-locate the
    element automatically if its id has gone stale since your last scan.
    """
    page = _session.page
    selector = f'[data-qa-agent-id="{element_id}"]'

    try:
        await page.click(selector, timeout=5000)
        return f"Clicked element with id={element_id}"
    except (PWTimeoutError, Exception) as first_error:
        new_id = await _relocate_element(element_description, action="click")
        if new_id is None:
            HEAL_LOG.append(
                {"action": "click", "description": element_description, "healed": False}
            )
            return (
                f"FAILED to click element id={element_id} (\"{element_description}\"): "
                f"{first_error}. Attempted self-heal but no matching element was found "
                f"on the current page."
            )
        try:
            await page.click(f'[data-qa-agent-id="{new_id}"]', timeout=5000)
            HEAL_LOG.append(
                {
                    "action": "click",
                    "description": element_description,
                    "healed": True,
                    "old_id": element_id,
                    "new_id": new_id,
                }
            )
            return (
                f"Original id={element_id} was stale. Self-healed: re-located "
                f"\"{element_description}\" as id={new_id} and clicked it successfully."
            )
        except Exception as heal_error:
            HEAL_LOG.append(
                {"action": "click", "description": element_description, "healed": False}
            )
            return (
                f"FAILED to click element (\"{element_description}\"): self-heal found "
                f"candidate id={new_id} but clicking it also failed: {heal_error}"
            )


@tool
async def fill_element(element_id: str, text: str, element_description: str) -> str:
    """
    Type the given text into the input/textarea with the given id
    (from get_page_elements). Clears any existing value first.
    Always also pass element_description: a short phrase describing the
    element (its placeholder, name, or label as shown by get_page_elements) --
    e.g. "Search input" or "placeholder='Email address'". This is used to
    re-locate the element automatically if its id has gone stale since your
    last scan.
    """
    page = _session.page
    selector = f'[data-qa-agent-id="{element_id}"]'

    try:
        await page.fill(selector, text, timeout=5000)
        return f"Filled element id={element_id} with text: {text}"
    except (PWTimeoutError, Exception) as first_error:
        new_id = await _relocate_element(element_description, action="fill")
        if new_id is None:
            HEAL_LOG.append(
                {"action": "fill", "description": element_description, "healed": False}
            )
            return (
                f"FAILED to fill element id={element_id} (\"{element_description}\"): "
                f"{first_error}. Attempted self-heal but no matching element was found "
                f"on the current page."
            )
        try:
            await page.fill(f'[data-qa-agent-id="{new_id}"]', text, timeout=5000)
            HEAL_LOG.append(
                {
                    "action": "fill",
                    "description": element_description,
                    "healed": True,
                    "old_id": element_id,
                    "new_id": new_id,
                }
            )
            return (
                f"Original id={element_id} was stale. Self-healed: re-located "
                f"\"{element_description}\" as id={new_id} and filled it with: {text}"
            )
        except Exception as heal_error:
            HEAL_LOG.append(
                {"action": "fill", "description": element_description, "healed": False}
            )
            return (
                f"FAILED to fill element (\"{element_description}\"): self-heal found "
                f"candidate id={new_id} but filling it also failed: {heal_error}"
            )


@tool
async def press_key(element_id: str, key: str, element_description: str) -> str:
    """
    Press a keyboard key while focused on the element with the given id
    (from get_page_elements). Use this to submit search boxes or forms that
    don't have a separate visible submit button -- key="Enter" is the most
    common case. Other valid values follow Playwright's key names, e.g.
    "Tab", "Escape".
    Always also pass element_description (same convention as click_element/
    fill_element) so the element can be re-located if it goes stale.
    """
    page = _session.page
    selector = f'[data-qa-agent-id="{element_id}"]'

    try:
        await page.press(selector, key, timeout=5000)
        return f"Pressed '{key}' on element id={element_id}"
    except (PWTimeoutError, Exception) as first_error:
        new_id = await _relocate_element(element_description, action=f"press '{key}' on")
        if new_id is None:
            HEAL_LOG.append(
                {"action": "press_key", "description": element_description, "healed": False}
            )
            return (
                f"FAILED to press '{key}' on element id={element_id} "
                f"(\"{element_description}\"): {first_error}. Attempted self-heal but no "
                f"matching element was found on the current page."
            )
        try:
            await page.press(f'[data-qa-agent-id="{new_id}"]', key, timeout=5000)
            HEAL_LOG.append(
                {
                    "action": "press_key",
                    "description": element_description,
                    "healed": True,
                    "old_id": element_id,
                    "new_id": new_id,
                }
            )
            return (
                f"Original id={element_id} was stale. Self-healed: re-located "
                f"\"{element_description}\" as id={new_id} and pressed '{key}' successfully."
            )
        except Exception as heal_error:
            HEAL_LOG.append(
                {"action": "press_key", "description": element_description, "healed": False}
            )
            return (
                f"FAILED to press '{key}' on element (\"{element_description}\"): self-heal "
                f"found candidate id={new_id} but pressing it also failed: {heal_error}"
            )


@tool
async def assert_text_present(expected_text: str) -> str:
    """
    Check whether the given text currently appears anywhere on the page.
    Use this as your final verification step to confirm the goal succeeded.
    """
    page = _session.page
    content = await page.content()
    found = expected_text.lower() in content.lower()
    return f"Text '{expected_text}' {'WAS found' if found else 'was NOT found'} on the page."


@tool
async def get_current_url() -> str:
    """Return the current page URL."""
    return _session.page.url


ALL_TOOLS = [
    navigate,
    get_page_elements,
    click_element,
    fill_element,
    press_key,
    assert_text_present,
    get_current_url,
]
