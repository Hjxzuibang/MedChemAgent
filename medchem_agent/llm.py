"""Thin OpenAI-compatible LLM client.

Connects directly (trust_env=False) so requests bypass any system proxy. The
API key is read from an environment variable (default ``OPENAI_API_KEY``) or,
optionally, from a JSON auth file (see ``llm.auth_file`` in config.yaml).
"""
import json
import os
import time

import httpx
from openai import OpenAI

from . import config


def _load_api_key(auth_file: str, api_key_env: str = "OPENAI_API_KEY") -> str:
    key = os.environ.get(api_key_env)
    if key:
        return key
    if auth_file:
        with open(auth_file, "r") as f:
            data = json.load(f)
        return data.get("OPENAI_API_KEY") or data.get("api_key") or data.get("token")
    return None


def make_client() -> OpenAI:
    cfg = config.load_config()["llm"]
    api_key_env = cfg.get("api_key_env", "OPENAI_API_KEY")
    key = _load_api_key(cfg.get("auth_file", ""), api_key_env)
    if not key:
        raise RuntimeError(
            f"No API key found: set the {api_key_env} environment variable "
            "or configure llm.auth_file in config.yaml.")
    http = httpx.Client(trust_env=False, timeout=cfg.get("timeout_sec", 180))
    return OpenAI(base_url=cfg["base_url"], api_key=key, http_client=http)


_CLIENT = None


def get_client() -> OpenAI:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = make_client()
    return _CLIENT


def chat(messages, model=None, temperature=None, max_tokens=None,
         tools=None, tool_choice="auto", max_retries=6):
    """Single chat-completion call with exponential backoff for transient errors.

    The gateway can intermittently return 503 under load; a bare retry loop fires
    every attempt within a second against an already-overloaded backend and always
    fails. Backoff lets the backend drain before the next attempt.
    """
    cfg = config.load_config()["llm"]
    client = get_client()
    kwargs = dict(
        model=model or cfg["model"],
        messages=messages,
        temperature=temperature if temperature is not None else cfg.get("temperature", 0.7),
        max_tokens=max_tokens or cfg.get("max_tokens", 3000),
    )
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice

    last_err = None
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:  # noqa: BLE001 - retry transient gateway errors
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(min(30.0, 2.0 ** attempt))  # 1,2,4,8,16,30 s
    raise last_err
