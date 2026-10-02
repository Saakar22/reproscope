"""LLM provider layer: dev mode, Groq/OpenAI-compatible requests, retries, errors, caching."""
from __future__ import annotations

import json

import httpx
import pytest

from reproscope import config, llm

SCHEMA = {"type": "object", "additionalProperties": False,
          "properties": {"answer": {"type": "integer"}}, "required": ["answer"]}


@pytest.fixture()
def live(monkeypatch):
    """Configure a fake live provider and capture requests through a mock transport."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_secret")
    monkeypatch.setenv("LLM_MAX_RETRIES", "2")
    config.override_settings(config.Settings())
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    calls: list[httpx.Request] = []
    responses: list[httpx.Response] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return responses.pop(0)

    llm.set_transport(httpx.MockTransport(handler))
    yield calls, responses
    llm.set_transport(None)


def ok(content: dict | str, finish="stop") -> httpx.Response:
    text = content if isinstance(content, str) else json.dumps(content)
    return httpx.Response(200, json={"model": "openai/gpt-oss-120b",
                                     "choices": [{"message": {"content": text}, "finish_reason": finish}],
                                     "usage": {"total_tokens": 42}})


def call(**kw):
    return llm.complete_json(name="t", schema=SCHEMA, system="sys", user=kw.pop("user", "q"),
                             dev_sample=lambda: {"answer": -1}, **kw)


def test_dev_mode_returns_labelled_sample():
    assert config.get_settings().llm_mode == "dev"
    result = call()
    assert result.origin == "dev_sample" and result.data == {"answer": -1}


def test_live_request_shape_and_result(live):
    calls, responses = live
    responses.append(ok({"answer": 7}))
    result = call()
    assert result.origin == "llm" and result.data == {"answer": 7} and not result.cached
    req = calls[0]
    assert str(req.url) == "https://api.groq.com/openai/v1/chat/completions"
    assert req.headers["authorization"] == "Bearer gsk_test_secret"
    body = json.loads(req.content)
    assert body["model"] == "openai/gpt-oss-120b"
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["response_format"]["json_schema"]["schema"] == SCHEMA
    assert body["temperature"] == 0


def test_cache_avoids_second_call_and_stores_no_key(live):
    calls, responses = live
    responses.append(ok({"answer": 1}))
    first, second = call(user="same"), call(user="same")
    assert len(calls) == 1 and second.cached and second.data == first.data
    for f in config.get_settings().llm_cache_dir.iterdir():
        assert "gsk_test_secret" not in f.read_text()


def test_retries_rate_limit_then_succeeds(live):
    calls, responses = live
    responses += [httpx.Response(429, headers={"retry-after": "1"}, json={"error": {"message": "slow down"}}),
                  ok({"answer": 3})]
    assert call().data == {"answer": 3}
    assert len(calls) == 2


def test_gives_up_after_retries(live):
    calls, responses = live
    responses += [httpx.Response(503, json={"error": {"message": "busy"}}) for _ in range(3)]
    with pytest.raises(llm.LLMError, match="after 3 attempts.*503"):
        call()


def test_auth_error_is_not_retried_and_hides_key(live):
    calls, responses = live
    responses.append(httpx.Response(401, json={"error": {"message": "Invalid API Key gsk_test_secret"}}))
    with pytest.raises(llm.LLMError) as exc:
        call()
    assert len(calls) == 1
    assert "gsk_test_secret" not in str(exc.value) and "GROQ_API_KEY" in str(exc.value)


@pytest.mark.parametrize("response,msg", [
    (ok("not json"), "valid JSON"),
    (ok("[1, 2]"), "not an object"),
    (ok({"answer": 1}, finish="length"), "cut off"),
])
def test_bad_responses_raise(live, response, msg):
    _calls, responses = live
    responses.append(response)
    with pytest.raises(llm.LLMError, match=msg):
        call(use_cache=False)


def test_best_effort_mode_uses_json_object(live, monkeypatch):
    calls, responses = live
    monkeypatch.setenv("LLM_STRICT_JSON", "false")
    config.override_settings(config.Settings())
    responses.append(ok({"answer": 5}))
    call()
    body = json.loads(calls[0].content)
    assert body["response_format"] == {"type": "json_object"}
    assert '"answer"' in body["messages"][0]["content"]          # schema described in the prompt


def test_ping_reports_dev_mode():
    assert llm.ping()["mode"] == "dev"
