from typing import Any

import tiktoken
from transformers import AutoTokenizer, LlamaTokenizer  # type: ignore

from llms.lm_config import get_minimax_model_config


class Tokenizer(object):
    def __init__(self, provider: str, model_name: str) -> None:
        self._uses_huggingface_tokenizer = False
        if provider == "openai":
            model_config = get_minimax_model_config(model_name)
            if model_config and model_config["tokenizer"] == "cl100k_base":
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
            elif model_config:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_config["tokenizer"]
                )
                self._uses_huggingface_tokenizer = True
            else:
                self.tokenizer = tiktoken.encoding_for_model(model_name)
        elif provider == "huggingface":
            self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
            # turn off adding special tokens automatically
            self.tokenizer.add_special_tokens = False  # type: ignore[attr-defined]
            self.tokenizer.add_bos_token = False  # type: ignore[attr-defined]
            self.tokenizer.add_eos_token = False  # type: ignore[attr-defined]
            self._uses_huggingface_tokenizer = True
        elif provider == "google":
            self.tokenizer = None  # Not used for input length computation, as Gemini is based on characters
        else:
            raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        if self._uses_huggingface_tokenizer:
            return self.tokenizer.encode(text, add_special_tokens=False)
        return self.tokenizer.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    def __call__(self, text: str) -> list[int]:
        return self.encode(text)
