"""
external_client.py — Universal OpenAI-compatible client for cloud LLM APIs.

Supports:
  - OpenAI (gpt-4o, gpt-4o-mini)
  - OpenRouter (qwen, deepseek, llama, claude proxies)
  - Groq (llama-3.3-70b-versatile, mixtral)
  - DeepSeek (deepseek-chat, deepseek-reasoner)
  - Any custom self-hosted or proxy endpoint (vLLM, LM Studio, etc.)
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

import requests

from config import config

logger = logging.getLogger(__name__)


def call_external_llm(
    messages: list[dict[str, str]],
    model: str | None = None,
    json_mode: bool = True,
) -> str | None:
    """
    Execute a chat completion request against an OpenAI-compatible REST endpoint.
    """
    if not config.LLM_API_KEY and config.LLM_PROVIDER not in ("custom",):
        logger.error(
            "External LLM provider '%s' is configured, but no LLM_API_KEY or OPENAI_API_KEY was found in environment.",
            config.LLM_PROVIDER,
        )
        return None

    model_name = model or config.ACTIVE_LLM_MODEL
    base_url = config.LLM_BASE_URL.rstrip("/")
    url = f"{base_url}/chat/completions"

    headers: dict[str, str] = {
        "Content-Type": "application/json",
    }
    if config.LLM_API_KEY:
        headers["Authorization"] = f"Bearer {config.LLM_API_KEY}"

    # Provider-specific headers
    if config.LLM_PROVIDER == "openrouter":
        headers["HTTP-Referer"] = "https://github.com/job-matcher"
        headers["X-Title"] = "AI Job Matcher"

    payload: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.2,
    }

    if json_mode:
        # Many providers support response_format: {"type": "json_object"}
        payload["response_format"] = {"type": "json_object"}

    try:
        logger.info(
            "Sending request to external LLM provider '%s' (%s, model: %s)...",
            config.LLM_PROVIDER,
            url,
            model_name,
        )
        resp = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=config.LLM_TIMEOUT,
        )

        # Some providers fail if response_format json_object is sent; retry without it if 400
        if resp.status_code == 400 and json_mode:
            logger.warning("Provider rejected response_format: json_object. Retrying without it...")
            payload.pop("response_format", None)
            resp = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=config.LLM_TIMEOUT,
            )

        resp.raise_for_status()
        data = resp.json()

        choices = data.get("choices", [])
        if not choices:
            logger.error("External LLM returned response without choices: %s", data)
            return None

        content = choices[0].get("message", {}).get("content", "")
        return content.strip()

    except requests.exceptions.ConnectionError as exc:
        logger.error("Failed to connect to external LLM endpoint at %s: %s", url, exc)
    except requests.exceptions.Timeout:
        logger.error("External LLM request timed out after %ds", config.LLM_TIMEOUT)
    except requests.exceptions.HTTPError as exc:
        logger.error("External LLM HTTP error (%s): %s", resp.status_code, resp.text)
    except Exception as exc:
        logger.error("Unexpected error querying external LLM: %s", exc)

    return None


def extract_json_from_text(raw_text: str) -> dict[str, Any] | None:
    """Extract and parse JSON object from raw LLM text output."""
    if not raw_text:
        return None

    # 1. Direct parse attempt
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    # 2. Markdown fenced block
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # 3. Outer bracket extraction
    outer_match = re.search(r"(\{.*\})", raw_text, re.DOTALL)
    if outer_match:
        try:
            return json.loads(outer_match.group(1))
        except json.JSONDecodeError:
            pass

    return None
