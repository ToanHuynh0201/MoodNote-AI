"""
Sinh dữ liệu nhật ký giả lập theo batch, ghi liên tục (checkpointing) để resume an toàn
khi phiên Colab bị ngắt giữa chừng.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from ...utils.config import load_config
from ...utils.emotion_constants import DEFAULT_EMOTION_LABELS
from ...utils.logger import get_logger
from .llm_client import BaseLLMClient
from .prompts import PROMPT_TEMPLATE_ID, build_prompt, sample_axis_values
from .schema import SyntheticSample, new_sample_id, now_iso

logger = get_logger("generate")


def _count_existing_samples_per_label(jsonl_path: Path) -> dict[int, int]:
    """Đếm số mẫu đã có theo nhãn trong file JSONL checkpoint (hỗ trợ resume)."""
    counts: dict[int, int] = dict.fromkeys(DEFAULT_EMOTION_LABELS, 0)
    if not jsonl_path.exists():
        return counts

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            counts[row["label"]] = counts.get(row["label"], 0) + 1

    return counts


def generate_dataset(
    client: BaseLLMClient,
    model_display_name: str,
    channel: str,
    output_path: str,
    generation_round: int,
    config_path: str = "configs/datagen_config.yaml",
    seed: int | None = None,
) -> Path:
    """
    Sinh tới `target_per_label` mẫu/nhãn cho MỘT model/kênh, resume từ dữ liệu đã có sẵn
    ở output_path (an toàn để chạy lại sau khi Colab bị ngắt phiên giữa chừng).

    Args:
        client: instance đã khởi tạo của BaseLLMClient (OpenRouterClient/HFLocalClient/
            ScriptedLLMClient)
        model_display_name: vd. "Llama-3-8B-Instruct" (lưu vào SyntheticSample.model)
        channel: "openrouter" | "hf_colab" | "scripted"
        output_path: File JSONL checkpoint (append mode, flush sau mỗi mẫu)
        generation_round: Số thứ tự đợt hiệu chỉnh prompt, lưu vào provenance
        config_path: Đường dẫn configs/datagen_config.yaml
        seed: Ghi đè seed trong config nếu có

    Returns:
        Path tới file JSONL đã ghi/append
    """
    config = load_config(config_path)
    rng = random.Random(seed if seed is not None else config["seed"])
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    existing = _count_existing_samples_per_label(out)
    gen_cfg = config["generation"]
    target_per_label = gen_cfg["target_per_label"]

    with open(out, "a", encoding="utf-8") as f:
        for label, label_name in DEFAULT_EMOTION_LABELS.items():
            deficit = target_per_label - existing.get(label, 0)
            if deficit <= 0:
                logger.info(
                    f"{label_name}: đã có {existing[label]}/{target_per_label}, bỏ qua."
                )
                continue

            for i in range(deficit):
                axes = sample_axis_values(config["diversity_axes"], rng)
                prompt = build_prompt(label, axes["style"], axes["length"], axes["context"])

                try:
                    response = client.generate(
                        prompt,
                        max_tokens=gen_cfg["max_new_tokens"],
                        temperature=gen_cfg["temperature"],
                        top_p=gen_cfg["top_p"],
                    )
                except Exception as e:
                    # Lỗi 1 mẫu không nên huỷ cả phiên sinh — Colab free tier rất quý thời gian.
                    logger.error(f"Sinh mẫu lỗi cho {label_name} (#{i}): {e}")
                    continue

                sample = SyntheticSample(
                    sample_id=new_sample_id(model_display_name, label),
                    text=response.text.strip(),
                    label=label,
                    label_name=label_name,
                    model=model_display_name,
                    channel=channel,
                    axis_style=axes["style"],
                    axis_length=axes["length"],
                    axis_context=axes["context"],
                    prompt_template_id=PROMPT_TEMPLATE_ID,
                    generation_round=generation_round,
                    generated_at=now_iso(),
                )
                f.write(sample.model_dump_json() + "\n")
                f.flush()

                if (i + 1) % gen_cfg["log_every_n_samples"] == 0:
                    logger.info(f"{label_name}: {i + 1}/{deficit} đã sinh ở đợt này.")

    return out
