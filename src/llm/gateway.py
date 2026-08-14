"""
LLM Gateway
-----------
Thin wrapper around OpenRouter so the rest of the agent never talks to a
specific provider directly. Mirrors the gateway pattern from interview-coach:
if you ever swap DeepSeek for another model, only this file changes.
"""

import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    """
    Returns a LangChain-compatible chat model pointed at OpenRouter.
    temperature=0.0 by default because for a QA agent we want deterministic,
    repeatable tool-calling behavior, not creative variation.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat")

    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY not set. Copy .env.example to .env and add your key."
        )

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        temperature=temperature,
        max_tokens=512
    )
