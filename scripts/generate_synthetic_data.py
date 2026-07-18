"""
Sinh dữ liệu nhật ký giả lập cho MỘT model/kênh.

Cách dùng:
    python scripts/generate_synthetic_data.py --model llama --channel scripted --round 1 \
        --output data/synthetic/raw/llama3_round1_dryrun.jsonl
    python scripts/generate_synthetic_data.py --model llama --channel openrouter --round 1 \
        --output data/synthetic/raw/llama3_round1.jsonl
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.synthetic.generate import generate_dataset  # noqa: E402
from src.data.synthetic.llm_client import (  # noqa: E402
    HFLocalClient,
    OpenRouterClient,
    ScriptedLLMClient,
)
from src.utils.config import load_config  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("generate_synthetic_data")

# Chỉ dùng cho --channel scripted (dry-run offline) — KHÔNG phải dữ liệu thật.
_DRYRUN_RESPONSES = [
    "Hôm nay là một ngày bình thường nhưng tôi thấy lòng mình nhẹ nhõm lạ.",
    "Tôi ngồi một mình trong phòng, nghĩ lại những chuyện đã xảy ra hôm nay.",
    "Mọi thứ diễn ra khá nhanh, tôi còn chưa kịp định thần để ghi lại hết cảm xúc.",
]


def _build_client(model: str, channel: str, config: dict):
    model_cfg = config["models"][model]

    if channel == "scripted":
        return ScriptedLLMClient(responses=_DRYRUN_RESPONSES, model=model_cfg["display_name"])

    if channel == "openrouter":
        openrouter_model_id = model_cfg["openrouter_model_id"]
        if openrouter_model_id is None:
            raise ValueError(
                f"Model '{model}' không có bản OpenRouter free-tier — dùng --channel hf_colab."
            )
        return OpenRouterClient(model_id=openrouter_model_id)

    if channel == "hf_colab":
        return HFLocalClient(model_id=model_cfg["bulk_model_id"])

    raise ValueError(f"Kênh không hợp lệ: {channel}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sinh dữ liệu nhật ký giả lập cho 1 model/kênh")
    parser.add_argument("--model", choices=["llama", "qwen"], required=True)
    parser.add_argument("--channel", choices=["openrouter", "hf_colab", "scripted"], required=True)
    parser.add_argument("--round", type=int, default=1, help="Số thứ tự đợt sinh (mặc định 1)")
    parser.add_argument("--output", required=True, help="File JSONL để ghi/append kết quả")
    parser.add_argument("--config", default="configs/datagen_config.yaml")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    client = _build_client(args.model, args.channel, config)
    model_display_name = config["models"][args.model]["display_name"]

    logger.info(
        f"Sinh dữ liệu: model={model_display_name} channel={args.channel} round={args.round}"
    )
    output_path = generate_dataset(
        client=client,
        model_display_name=model_display_name,
        channel=args.channel,
        output_path=args.output,
        generation_round=args.round,
        config_path=args.config,
        seed=args.seed,
    )
    logger.info(f"Hoàn tất, kết quả ghi tại {output_path}")


if __name__ == "__main__":
    main()
