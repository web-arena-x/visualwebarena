import asyncio
import json
from argparse import Namespace

import aiolimiter
import httpx
import openai
import pytest

from llms import call_llm
import llms.providers.openai_utils as openai_utils
import llms.tokenizers as tokenizers
from llms.lm_config import (
    MINIMAX_MODEL_CONFIGS,
    construct_llm_config,
    supports_multimodal_input,
)
from llms.tokenizers import Tokenizer


def _args(model: str, provider: str = "openai", context_length: int = 0) -> Namespace:
    return Namespace(
        provider=provider,
        model=model,
        mode="chat",
        temperature=1.0,
        top_p=0.9,
        context_length=context_length,
        max_tokens=384,
        stop_token=None,
        max_obs_length=3840,
        max_retry=1,
        model_endpoint=None,
    )


def test_active_catalog_is_flash_first() -> None:
    assert list(MINIMAX_MODEL_CONFIGS) == ["MiniMax-M3.1-Flash", "MiniMax-M3"]


@pytest.mark.parametrize("model", list(MINIMAX_MODEL_CONFIGS))
def test_minimax_model_metadata_is_applied(model) -> None:
    m3_config = construct_llm_config(_args(model))
    assert m3_config.gen_config["context_length"] == 1_000_000
    assert m3_config.gen_config["input_modalities"] == (
        "text",
        "image",
        "video",
    )
    assert m3_config.model == model
    assert supports_multimodal_input(model)

    m27_config = construct_llm_config(_args("MiniMax-M2.7"))
    assert m27_config.gen_config["context_length"] == 204_800
    assert m27_config.gen_config["input_modalities"] == ("text",)
    assert not supports_multimodal_input("MiniMax-M2.7")


@pytest.mark.parametrize("model", ["MiniMax-M3.1-Flash", "MiniMax-M3", "MiniMax-M2.7"])
def test_explicit_context_length_takes_precedence(model) -> None:
    config = construct_llm_config(_args(model, context_length=4096))
    assert config.gen_config["context_length"] == 4096


@pytest.mark.parametrize("model", ["MiniMax-M3", "MiniMax-M2.7"])
def test_minimax_uses_its_model_tokenizer(monkeypatch, model) -> None:
    calls = []

    class FakeTokenizer:
        def encode(self, text, **kwargs):
            calls.append((text, kwargs))
            return [1, 2]

        def decode(self, ids):
            return "decoded"

    def load_tokenizer(model_name):
        calls.append(("model", model_name))
        return FakeTokenizer()

    monkeypatch.setattr(tokenizers.AutoTokenizer, "from_pretrained", load_tokenizer)
    tokenizer = Tokenizer("openai", model)

    assert tokenizer.encode("hello") == [1, 2]
    assert tokenizer("world") == [1, 2]
    assert calls[0] == ("model", f"MiniMaxAI/{model}")
    assert calls[1:] == [
        ("hello", {"add_special_tokens": False}),
        ("world", {"add_special_tokens": False}),
    ]


def test_minimax_endpoint_selection(monkeypatch) -> None:
    created = []

    class FakeClient:
        def __init__(self, **kwargs):
            created.append(kwargs)

    monkeypatch.setattr(openai_utils, "OpenAI", FakeClient)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("MINIMAX_BASE_URL", raising=False)
    monkeypatch.setenv("MINIMAX_REGION", "cn_zh")

    openai_utils._get_client("MiniMax-M3")
    assert created[-1] == {
        "api_key": "test-openai-key",
        "base_url": "https://api.minimax.cn/v1",
    }

    monkeypatch.setenv("MINIMAX_BASE_URL", "https://proxy.example/v1")
    openai_utils._get_client("MiniMax-M2.7")
    assert created[-1]["base_url"] == "https://proxy.example/v1"


def test_other_provider_does_not_receive_minimax_metadata() -> None:
    config = construct_llm_config(_args("MiniMax-M3.1-Flash", provider="google"))
    assert config.provider == "google"
    assert config.model == "MiniMax-M3.1-Flash"
    assert config.gen_config["context_length"] == 0
    assert "input_modalities" not in config.gen_config
    assert "tokenizer" not in config.gen_config


def test_flash_uses_explicit_approximate_tokenizer(monkeypatch) -> None:
    class FakeEncoding:
        def encode(self, text):
            return [1, 2]

        def decode(self, ids):
            return "hello"

    calls = []

    def get_encoding(name):
        calls.append(name)
        return FakeEncoding()

    monkeypatch.setattr(tokenizers.tiktoken, "get_encoding", get_encoding)
    tokenizer = Tokenizer("openai", "MiniMax-M3.1-Flash")
    assert tokenizer.encode("hello") == tokenizer("hello") == [1, 2]
    assert tokenizer.decode([1, 2]) == "hello"
    assert calls == ["cl100k_base"]


@pytest.fixture
def sdk_transport(monkeypatch):
    for name in (
        "MINIMAX_API_KEY", "OPENAI_API_KEY", "MINIMAX_REGION",
        "MINIMAX_BASE_URL", "OPENAI_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MINIMAX_API_KEY", "test-minimax-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={
            "id": "test-completion", "object": "chat.completion", "created": 0,
            "model": json.loads(request.content)["model"],
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": "answer",
            }}],
        })

    transport = httpx.MockTransport(respond)
    real_client = openai.OpenAI
    real_async_client = openai.AsyncOpenAI
    clients = []

    def create_client(**kwargs):
        client = real_client(http_client=httpx.Client(transport=transport), **kwargs)
        clients.append(client)
        return client

    def create_async_client(**kwargs):
        client = real_async_client(
            http_client=httpx.AsyncClient(transport=transport), **kwargs
        )
        clients.append(client)
        return client

    monkeypatch.setattr(openai_utils, "OpenAI", create_client)
    monkeypatch.setattr(openai_utils, "AsyncOpenAI", create_async_client)
    yield requests
    for client in clients:
        if isinstance(client, real_async_client):
            asyncio.run(client.close())
        else:
            client.close()


@pytest.mark.parametrize("model", [
    "MiniMax-M3.1-Flash", "MiniMax-M3", "MiniMax-M2.7",
])
@pytest.mark.parametrize("region,host", [
    (None, "api.minimax.io"), ("cn_zh", "api.minimax.cn"),
])
def test_call_llm_serializes_exact_model_and_history(
    sdk_transport, monkeypatch, model, region, host
):
    if region:
        monkeypatch.setenv("MINIMAX_REGION", region)
    messages = [
        {"role": "system", "content": "Be helpful."},
        {"role": "assistant", "content": "<think>previous reasoning</think>",
         "reasoning_content": "previous reasoning",
         "tool_calls": [{"id": "call_1", "type": "function", "function": {
             "name": "lookup", "arguments": "{}",
         }}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "result"},
        {"role": "user", "content": "Continue."},
    ]
    if model != "MiniMax-M2.7":
        messages[-1]["content"] = [
            {"type": "text", "text": "Describe these."},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
            {"type": "video_url", "video_url": {"url": "mm_file://test-video"}},
        ]
    config = construct_llm_config(_args(model, context_length=4096))
    assert config.gen_config["context_length"] == 4096
    assert call_llm(config, messages) == "answer"
    request, = sdk_transport
    assert str(request.url) == f"https://{host}/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer test-minimax-key"
    # Omitted thinking/effort uses the API defaults (adaptive/max for Flash).
    # No unsupported service tier or reasoning_split=false is introduced.
    assert json.loads(request.content) == {
        "model": model, "messages": messages,
        "temperature": 1.0, "top_p": 0.9, "max_tokens": 384,
    }


@pytest.mark.parametrize("model,override,expected_key", [
    ("MiniMax-M3.1-Flash", "MINIMAX_BASE_URL", "test-minimax-key"),
    ("MiniMax-M3", "OPENAI_BASE_URL", "test-minimax-key"),
    ("MiniMax-M2.7", "MINIMAX_BASE_URL", "test-minimax-key"),
    ("custom-saved-model", "OPENAI_BASE_URL", "test-openai-key"),
    ("gpt-3.5-turbo-0613", "OPENAI_BASE_URL", "test-openai-key"),
])
def test_custom_endpoint_and_saved_models_are_preserved(
    sdk_transport, monkeypatch, model, override, expected_key
):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://fallback.example/v1")
    monkeypatch.setenv(override, "https://proxy.example/v1")
    config = construct_llm_config(_args(model))
    assert call_llm(config, [{"role": "user", "content": "Hello"}]) == "answer"
    request, = sdk_transport
    assert str(request.url) == "https://proxy.example/v1/chat/completions"
    assert json.loads(request.content)["model"] == model
    assert request.headers["authorization"] == f"Bearer {expected_key}"


def test_openai_default_ignores_minimax_settings(sdk_transport, monkeypatch):
    monkeypatch.setenv("MINIMAX_BASE_URL", "https://minimax-proxy.example/v1")
    monkeypatch.setenv("MINIMAX_REGION", "invalid-region")
    config = construct_llm_config(_args("gpt-3.5-turbo-0613"))
    assert config.gen_config["context_length"] == 0
    assert "input_modalities" not in config.gen_config
    assert call_llm(config, [{"role": "user", "content": "Hello"}]) == "answer"
    request, = sdk_transport
    assert str(request.url) == "https://api.openai.com/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer test-openai-key"


def test_explicit_legacy_endpoint_is_preserved(sdk_transport, monkeypatch):
    monkeypatch.setenv("MINIMAX_BASE_URL", "https://api.minimaxi.com/v1")
    monkeypatch.setenv("MINIMAX_REGION", "cn_zh")
    config = construct_llm_config(_args("MiniMax-M2.7"))
    assert call_llm(config, [{"role": "user", "content": "Hello"}]) == "answer"
    request, = sdk_transport
    assert str(request.url) == "https://api.minimaxi.com/v1/chat/completions"
    assert json.loads(request.content)["model"] == "MiniMax-M2.7"


def test_flash_falls_back_to_openai_key(sdk_transport, monkeypatch):
    monkeypatch.delenv("MINIMAX_API_KEY")
    config = construct_llm_config(_args("MiniMax-M3.1-Flash"))
    assert call_llm(config, [{"role": "user", "content": "Hello"}]) == "answer"
    assert sdk_transport[0].headers["authorization"] == "Bearer test-openai-key"


def test_invalid_minimax_region_fails_before_http(sdk_transport, monkeypatch):
    monkeypatch.setenv("MINIMAX_REGION", "invalid-region")
    config = construct_llm_config(_args("MiniMax-M3.1-Flash"))
    with pytest.raises(ValueError, match="Unsupported MiniMax region"):
        call_llm(config, [{"role": "user", "content": "Hello"}])
    assert sdk_transport == []


@pytest.mark.parametrize("model", ["MiniMax-M3.1-Flash", "MiniMax-M3", "MiniMax-M2.7"])
def test_async_sdk_serialization(sdk_transport, model):
    messages = [{"role": "user", "content": "Hello"}]

    async def generate():
        return await openai_utils._throttled_openai_chat_completion_acreate(
            model, messages, 1.0, 384, 0.9, aiolimiter.AsyncLimiter(300)
        )

    response = asyncio.run(generate())
    assert response.choices[0].message.content == "answer"
    request, = sdk_transport
    assert str(request.url) == "https://api.minimax.io/v1/chat/completions"
    assert json.loads(request.content) == {
        "model": model, "messages": messages,
        "temperature": 1.0, "max_tokens": 384, "top_p": 0.9,
    }
