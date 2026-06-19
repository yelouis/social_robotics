"""Single-turn ollama `/api/chat` call with an ENFORCED request timeout.

Why this exists: the `ollama` Python client passes `timeout=None` to httpx on
every request, so its `Client(timeout=...)` is silently ignored and a wedged
generation (e.g. a vision-prefill deadlock seen on qwen2.5vl) hangs the caller
*forever*. The supervised runner only relaunches on crashes, not hangs, so one
stuck VLM call stalls an entire unattended batch.

Calling the HTTP API directly with httpx makes the timeout real: on timeout
httpx raises `httpx.TimeoutException` **and closes the socket**, so ollama sees
the client disconnect and cancels the generation — freeing the model runner for
the next request. Callers (the social-presence gate, 03b, climax) already wrap
the call in try/except and fall back to a default, so a raised timeout is
handled gracefully (the clip is skipped/defaulted, not stuck).
"""
import base64
import os

import httpx


def _default_host() -> str:
    h = (os.getenv("OLLAMA_HOST") or "127.0.0.1:11434").strip()
    if not h.startswith(("http://", "https://")):
        h = "http://" + h
    return h.rstrip("/")


def ollama_chat(model, prompt, image_paths=None, options=None, fmt=None,
                timeout=180.0, host=None) -> str:
    """POST one user message to ollama `/api/chat` (non-streaming) and return the
    assistant message content. ``image_paths`` are read and base64-encoded.
    ``fmt`` maps to the API's ``format`` field (e.g. "json"). Raises
    ``httpx.TimeoutException`` if the call exceeds ``timeout`` seconds (the
    connection is closed, so the server-side generation is cancelled)."""
    base = host or _default_host()
    message = {"role": "user", "content": prompt}
    if image_paths:
        message["images"] = [
            base64.b64encode(open(p, "rb").read()).decode("ascii") for p in image_paths
        ]
    payload = {"model": model, "messages": [message], "stream": False}
    if options:
        payload["options"] = options
    if fmt:
        payload["format"] = fmt
    resp = httpx.post(f"{base}/api/chat", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json().get("message", {}).get("content", "")
