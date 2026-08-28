"""
LLM client cho sinh dữ liệu nhật ký giả lập.

Hai kênh thật (theo hạ tầng đã chốt):
- OpenRouterClient: REST free-tier, tương thích OpenAI — dùng thử prompt cho Llama.
- HFLocalClient: transformers + 4-bit quantization — sinh hàng loạt trên Colab, cả 2 model.
Cộng thêm ScriptedLLMClient: trả lời theo kịch bản cố định, dùng cho test/dry-run offline,
KHÔNG dùng để sinh dữ liệu thật.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Protocol

import httpx

from ...utils.logger import get_logger

logger = get_logger("llm_client")

DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass
class LLMResponse:
    text: str
    model: str
    raw: dict | None = None


class BaseLLMClient(Protocol):
    def generate(
        self, prompt: str, max_tokens: int = 220, temperature: float = 0.9, top_p: float = 0.95
    ) -> LLMResponse: ...


class OpenRouterClient:
    """Client REST cho OpenRouter free-tier (tương thích API OpenAI)."""

    def __init__(
        self,
        model_id: str,
        api_key: str | None = None,
        base_url: str = DEFAULT_OPENROUTER_BASE_URL,
        timeout: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """
        Args:
            model_id: OpenRouter model id (vd. "meta-llama/llama-3.1-8b-instruct:free")
            api_key: API key; mặc định đọc từ env OPENROUTER_API_KEY
            base_url: Base URL của OpenRouter API
            timeout: Timeout (giây) cho mỗi request
            transport: httpx transport tùy chỉnh (dùng để test qua httpx.MockTransport)
        """
        self.model_id = model_id
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self._client = httpx.Client(base_url=base_url, transport=transport, timeout=timeout)

    def generate(
        self, prompt: str, max_tokens: int = 220, temperature: float = 0.9, top_p: float = 0.95
    ) -> LLMResponse:
        try:
            response = self._client.post(
                "/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                },
            )
            response.raise_for_status()
            payload = response.json()
            return LLMResponse(
                text=payload["choices"][0]["message"]["content"], model=self.model_id, raw=payload
            )
        except Exception as e:
            logger.error(f"OpenRouter generate() failed: {e}")
            raise


def _import_hf_stack():
    """Seam riêng để test có thể monkeypatch, mô phỏng trường hợp thiếu transformers."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    return AutoModelForCausalLM, AutoTokenizer


class HFLocalClient:
    """
    Client HuggingFace 4-bit — chỉ chạy trên Colab GPU thật (T4 16GB).

    Chưa được xác nhận chạy thật trong phiên code này (không có GPU); phần load model
    4-bit cần owner tự xác nhận khi chạy trên Colab. Hỗ trợ dependency-injection qua
    `model=`/`tokenizer=` để test không cần tải model/transformers thật.
    """

    def __init__(
        self,
        model_id: str,
        model=None,
        tokenizer=None,
        load_in_4bit: bool = True,
        device_map: str = "auto",
        hf_token: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.load_in_4bit = load_in_4bit

        if model is not None and tokenizer is not None:
            # Đường dependency-injection — dùng trong test, không đụng transformers/torch.
            self.model = model
            self.tokenizer = tokenizer
            return

        try:
            auto_model_cls, auto_tokenizer_cls = _import_hf_stack()
        except ImportError as exc:
            raise ImportError(
                "Cần transformers/torch để dùng HFLocalClient thật. "
                "Cài bằng: pip install -r requirements.txt (chạy trên môi trường Colab/GPU)."
            ) from exc

        token = hf_token or os.getenv("HF_TOKEN")
        self.tokenizer = auto_tokenizer_cls.from_pretrained(model_id, token=token)

        quantization_config = None
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(load_in_4bit=True)

        self.model = auto_model_cls.from_pretrained(
            model_id,
            token=token,
            quantization_config=quantization_config,
            device_map=device_map,
        )

    def generate(
        self, prompt: str, max_tokens: int = 220, temperature: float = 0.9, top_p: float = 0.95
    ) -> LLMResponse:
        messages = [{"role": "user", "content": prompt}]
        # enable_thinking=False: Qwen3 mặc định sinh <think>...</think> trước câu trả lời,
        # ăn hết max_new_tokens trước khi tới nội dung; template của model khác bỏ qua cờ này.
        encoded = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            enable_thinking=False,
        ).to(self.model.device)
        input_ids = encoded["input_ids"]
        # temperature=0.0 nghĩa là greedy decoding (cross-LLM audit cần review tất định).
        # transformers từ chối temperature=0.0 khi do_sample=True, và bỏ qua kèm warning
        # temperature/top_p khi do_sample=False — nên chỉ truyền chúng ở nhánh sampling.
        sampling_kwargs = (
            {"do_sample": True, "temperature": temperature, "top_p": top_p}
            if temperature > 0
            else {"do_sample": False}
        )
        output_ids = self.model.generate(
            **encoded,
            max_new_tokens=max_tokens,
            **sampling_kwargs,
        )
        generated = output_ids[0][input_ids.shape[-1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        # enable_thinking=False chỉ là bias mạnh, không hard-block — model reasoning (vd. Qwen3)
        # đôi khi vẫn tự mở lại <think>...</think>. Cắt bỏ nếu có, giữ nguyên nếu không.
        text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()
        return LLMResponse(text=text, model=self.model_id, raw=None)


class ScriptedLLMClient:
    """Trả lời tuần hoàn theo danh sách kịch bản. Chỉ dùng cho test + dry-run offline."""

    def __init__(self, responses: list[str], model: str = "scripted") -> None:
        self._responses = responses
        self.model = model
        self.calls: list[str] = []

    def generate(
        self, prompt: str, max_tokens: int = 220, temperature: float = 0.9, top_p: float = 0.95
    ) -> LLMResponse:
        self.calls.append(prompt)
        text = self._responses[(len(self.calls) - 1) % len(self._responses)]
        return LLMResponse(text=text, model=self.model, raw=None)
