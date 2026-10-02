"""Provider-agnostic structured LLM calls.

Every call asks for JSON matching a schema and returns an `LLMResult` that records
where the data came from:

* ``origin="llm"``         a real provider response (Groq or any OpenAI-compatible API)
* ``origin="dev_sample"``  no API key configured: the caller's deterministic sample,
                           which the UI and reports always label as such

Responses are cached on disk keyed by a hash of the full request, so replays and
re-runs of a finished analysis don't need the network. The API key is never logged
or stored in the cache.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import httpx

from .config import Settings, get_settings


class LLMError(RuntimeError):
    """The provider call failed in a way the user should see (bad key, model, quota...)."""


@dataclass
class LLMResult:
    data: dict[str, Any]
    origin: str                    # "llm" | "dev_sample"
    model: Optional[str] = None
    cached: bool = False
    usage: dict[str, Any] = field(default_factory=dict)


# Shared transport hook so tests can inject httpx.MockTransport.
_transport: Optional[httpx.BaseTransport] = None


def set_transport(transport: Optional[httpx.BaseTransport]) -> None:
    global _transport
    _transport = transport


def _cache_key(settings: Settings, name: str, schema: dict, system: str, user: str) -> str:
    blob = json.dumps({"provider": settings.llm_provider, "base_url": settings.llm_base_url,
                       "model": settings.llm_model, "strict": settings.llm_strict_json,
                       "name": name, "schema": schema, "system": system, "user": user},
                      sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def _redact(text: str, secret: Optional[str]) -> str:
    return text.replace(secret, "***") if secret else text


def complete_json(*, name: str, schema: dict[str, Any], system: str, user: str,
                  dev_sample: Callable[[], dict[str, Any]], max_tokens: int = 8000,
                  use_cache: bool = True) -> LLMResult:
    """Ask the configured model for JSON conforming to `schema`.

    `dev_sample` is only used when no provider key is configured. Callers must still
    validate the returned data (pydantic) — strict decoding is not a substitute.
    """
    settings = get_settings()
    if settings.llm_mode == "dev":
        return LLMResult(data=dev_sample(), origin="dev_sample")

    key = _cache_key(settings, name, schema, system, user)
    cache_file = settings.llm_cache_dir / f"{key}.json"
    if use_cache and cache_file.is_file():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        return LLMResult(data=cached["data"], origin="llm", model=cached.get("model"), cached=True,
                         usage=cached.get("usage", {}))

    data, model, usage = _call_openai_compatible(settings, name, schema, system, user, max_tokens)
    settings.llm_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps({"name": name, "model": model, "usage": usage, "data": data},
                                     indent=1), encoding="utf-8")
    return LLMResult(data=data, origin="llm", model=model, usage=usage)


def _call_openai_compatible(settings: Settings, name: str, schema: dict, system: str, user: str,
                            max_tokens: int) -> tuple[dict, str, dict]:
    if settings.llm_strict_json:
        response_format = {"type": "json_schema",
                           "json_schema": {"name": name, "strict": True, "schema": schema}}
    else:  # best effort: describe the schema in the prompt and ask for a JSON object
        response_format = {"type": "json_object"}
        system = f"{system}\n\nRespond with one JSON object matching this JSON Schema:\n{json.dumps(schema)}"
    body = {
        "model": settings.llm_model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "response_format": response_format,
        "temperature": 0,
        "max_completion_tokens": max_tokens,
    }
    headers = {"Authorization": f"Bearer {settings.llm_api_key}", "Content-Type": "application/json"}
    url = f"{settings.llm_base_url}/chat/completions"

    last_error = "unknown error"
    with httpx.Client(timeout=settings.llm_timeout_s, transport=_transport) as client:
        for attempt in range(settings.llm_max_retries + 1):
            try:
                resp = client.post(url, json=body, headers=headers)
            except httpx.HTTPError as exc:
                last_error = f"network error: {type(exc).__name__}"
                _sleep_backoff(attempt, None)
                continue
            if resp.status_code == 200:
                return _parse_response(resp, settings)
            detail = _error_detail(resp)
            if resp.status_code in (429, 500, 502, 503, 504):
                last_error = f"HTTP {resp.status_code}: {detail}"
                _sleep_backoff(attempt, resp.headers.get("retry-after"))
                continue
            # 400/401/403/404/413: retrying won't help
            hint = {401: " (check GROQ_API_KEY / LLM_API_KEY)",
                    404: f" (check LLM_MODEL='{settings.llm_model}' and LLM_BASE_URL)",
                    413: " (request too large for this model's limits)"}.get(resp.status_code, "")
            raise LLMError(_redact(f"LLM request failed: HTTP {resp.status_code}: {detail}{hint}",
                                   settings.llm_api_key))
    raise LLMError(_redact(f"LLM request failed after {settings.llm_max_retries + 1} attempts: {last_error}",
                           settings.llm_api_key))


def _parse_response(resp: httpx.Response, settings: Settings) -> tuple[dict, str, dict]:
    payload = resp.json()
    try:
        choice = payload["choices"][0]
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError("LLM response had no message content") from exc
    if choice.get("finish_reason") == "length":
        raise LLMError("LLM response was cut off (max tokens reached); try fewer claims or pages")
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError) as exc:
        raise LLMError("LLM did not return valid JSON") from exc
    if not isinstance(data, dict):
        raise LLMError("LLM returned JSON that is not an object")
    return data, payload.get("model", settings.llm_model), payload.get("usage", {})


def _error_detail(resp: httpx.Response) -> str:
    try:
        err = resp.json().get("error", {})
        return str(err.get("message") or err)[:300]
    except Exception:
        return resp.text[:300]


def _sleep_backoff(attempt: int, retry_after: Optional[str]) -> None:
    try:
        delay = float(retry_after) if retry_after is not None else 2 ** attempt
    except ValueError:
        delay = 2 ** attempt
    time.sleep(min(delay, 30))


def ping() -> dict[str, Any]:
    """Tiny live call used by the health check button. Never cached."""
    settings = get_settings()
    if settings.llm_mode == "dev":
        return {"ok": False, "mode": "dev", "detail": "No API key configured (GROQ_API_KEY or LLM_API_KEY)."}
    schema = {"type": "object", "additionalProperties": False,
              "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
    t0 = time.monotonic()
    try:
        result = complete_json(name="ping", schema=schema, system="Reply with JSON.",
                               user='Return {"ok": true}.', dev_sample=lambda: {"ok": True},
                               max_tokens=1000, use_cache=False)  # reasoning models need headroom
    except LLMError as exc:
        return {"ok": False, "mode": "live", "detail": str(exc)}
    return {"ok": bool(result.data.get("ok")), "mode": "live", "model": result.model,
            "seconds": round(time.monotonic() - t0, 2)}
