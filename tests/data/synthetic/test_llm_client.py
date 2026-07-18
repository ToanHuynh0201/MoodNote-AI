"""Tests for LLM clients (network-free: OpenRouter via httpx.MockTransport, HFLocalClient via DI)."""

import json

import httpx
import pytest

from src.data.synthetic.llm_client import HFLocalClient, OpenRouterClient, ScriptedLLMClient


class _FakeTensor:
    """Stands in for a torch tensor — just enough surface for HFLocalClient.generate()."""

    def __init__(self, data: list[list[int]]) -> None:
        self.data = data
        self.shape = (len(data), len(data[0]) if data else 0)

    def __getitem__(self, idx):
        return self.data[idx]


class _FakeEncoding(dict):
    """Stands in for a transformers BatchEncoding — dict-like plus `.to(device)`."""

    def to(self, device):
        return self


class _FakeTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt, return_tensors, return_dict):
        self.last_messages = messages
        return _FakeEncoding(input_ids=_FakeTensor([[1, 2, 3]]))

    def decode(self, ids, skip_special_tokens):
        return "generated diary text"


class _FakeModel:
    device = "cpu"

    def generate(self, input_ids, max_new_tokens, temperature, top_p, do_sample):
        return _FakeTensor([[1, 2, 3, 4, 5, 6]])


def test_openrouter_client_generate_parses_response_content():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "sample diary"}}]})

    client = OpenRouterClient(
        model_id="meta-llama/llama-3.1-8b-instruct:free",
        api_key="test-key",
        transport=httpx.MockTransport(handler),
    )

    response = client.generate("prompt text")

    assert response.text == "sample diary"
    assert response.model == "meta-llama/llama-3.1-8b-instruct:free"


def test_openrouter_client_reads_api_key_from_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key-123")
    seen_auth = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_auth["value"] = request.headers.get("authorization")
        return httpx.Response(200, json={"choices": [{"message": {"content": "x"}}]})

    client = OpenRouterClient(model_id="m", transport=httpx.MockTransport(handler))
    client.generate("prompt")

    assert seen_auth["value"] == "Bearer env-key-123"


def test_openrouter_client_raises_and_logs_on_http_error(caplog):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    client = OpenRouterClient(model_id="m", api_key="k", transport=httpx.MockTransport(handler))

    with caplog.at_level("ERROR"), pytest.raises(httpx.HTTPStatusError):
        client.generate("prompt")

    assert "OpenRouter generate() failed" in caplog.text


def test_openrouter_client_sends_expected_request_payload():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    client = OpenRouterClient(
        model_id="model-x", api_key="k", transport=httpx.MockTransport(handler)
    )
    client.generate("hello", max_tokens=50, temperature=0.5, top_p=0.8)

    payload = captured["payload"]
    assert payload["model"] == "model-x"
    assert payload["max_tokens"] == 50
    assert payload["temperature"] == 0.5
    assert payload["top_p"] == 0.8
    assert payload["messages"] == [{"role": "user", "content": "hello"}]


def test_hf_local_client_uses_injected_model_and_tokenizer_without_importing_transformers(
    monkeypatch,
):
    import src.data.synthetic.llm_client as llm_client_module

    def _raise_if_called():
        raise AssertionError("_import_hf_stack should not be called when DI'd")

    monkeypatch.setattr(llm_client_module, "_import_hf_stack", _raise_if_called)

    client = HFLocalClient(model_id="some/model", model=_FakeModel(), tokenizer=_FakeTokenizer())
    response = client.generate("prompt")

    assert response.text == "generated diary text"
    assert response.model == "some/model"


def test_hf_local_client_raises_clear_import_error_without_injection(monkeypatch):
    import src.data.synthetic.llm_client as llm_client_module

    def _raise_import_error():
        raise ImportError("no transformers installed")

    monkeypatch.setattr(llm_client_module, "_import_hf_stack", _raise_import_error)

    with pytest.raises(ImportError, match="Cần transformers/torch"):
        HFLocalClient(model_id="some/model")


def test_scripted_llm_client_cycles_through_responses():
    client = ScriptedLLMClient(responses=["a", "b", "c"])

    texts = [client.generate("prompt").text for _ in range(5)]

    assert texts == ["a", "b", "c", "a", "b"]
    assert client.calls == ["prompt"] * 5
