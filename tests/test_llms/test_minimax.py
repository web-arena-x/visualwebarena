from argparse import Namespace

import llms.providers.openai_utils as openai_utils
import llms.tokenizers as tokenizers
from llms.lm_config import construct_llm_config, supports_multimodal_input
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


def test_minimax_model_metadata_is_applied() -> None:
    m3_config = construct_llm_config(_args("MiniMax-M3"))
    assert m3_config.gen_config["context_length"] == 1_000_000
    assert m3_config.gen_config["input_modalities"] == (
        "text",
        "image",
        "video",
    )
    assert supports_multimodal_input("MiniMax-M3")

    m27_config = construct_llm_config(_args("MiniMax-M2.7"))
    assert m27_config.gen_config["context_length"] == 204_800
    assert m27_config.gen_config["input_modalities"] == ("text",)
    assert not supports_multimodal_input("MiniMax-M2.7")


def test_explicit_context_length_takes_precedence() -> None:
    config = construct_llm_config(_args("MiniMax-M3", context_length=4096))
    assert config.gen_config["context_length"] == 4096


def test_minimax_uses_its_model_tokenizer(monkeypatch) -> None:
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
    tokenizer = Tokenizer("openai", "MiniMax-M3")

    assert tokenizer.encode("hello") == [1, 2]
    assert tokenizer("world") == [1, 2]
    assert calls[0] == ("model", "MiniMaxAI/MiniMax-M3")
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
        "base_url": "https://api.minimaxi.com/v1",
    }

    monkeypatch.setenv("MINIMAX_BASE_URL", "https://proxy.example/v1")
    openai_utils._get_client("MiniMax-M2.7")
    assert created[-1]["base_url"] == "https://proxy.example/v1"
