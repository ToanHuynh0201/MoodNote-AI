"""
Chạy cross-LLM audit: Qwen review mẫu do Llama sinh, và ngược lại.

Cách dùng:
    python scripts/cross_llm_audit.py --input data/synthetic/leakage_checked/clean.jsonl \
        --llama-channel scripted --qwen-channel scripted \
        --output data/synthetic/qa/cross_llm/cross_llm_review.jsonl
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.synthetic.llm_client import (  # noqa: E402
    HFLocalClient,
    OpenRouterClient,
    ScriptedLLMClient,
)
from src.data.synthetic.schema import read_samples_jsonl  # noqa: E402
from src.qa.cross_llm_check import run_cross_llm_audit, write_reviews_jsonl  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("cross_llm_audit")

# Chỉ dùng cho --channel scripted (dry-run offline) — KHÔNG phải audit thật.
_DRYRUN_REVIEW_RESPONSES = ["NHÃN: Enjoyment\nTỰ_NHIÊN: có"]


def _build_reviewer_client(model: str, channel: str, config: dict):
    model_cfg = config["models"][model]

    if channel == "scripted":
        return ScriptedLLMClient(
            responses=_DRYRUN_REVIEW_RESPONSES, model=model_cfg["display_name"]
        )

    if channel == "openrouter":
        openrouter_model_id = model_cfg["openrouter_model_id"]
        if openrouter_model_id is None:
            raise ValueError(f"Model '{model}' không có bản OpenRouter free-tier.")
        return OpenRouterClient(model_id=openrouter_model_id)

    if channel == "hf_colab":
        return HFLocalClient(model_id=model_cfg["bulk_model_id"])

    raise ValueError(f"Kênh không hợp lệ: {channel}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy cross-LLM audit")
    parser.add_argument("--input", required=True, help="File JSONL đầu vào (đã qua leakage_guard)")
    parser.add_argument("--output", default="data/synthetic/qa/cross_llm/cross_llm_review.jsonl")
    parser.add_argument("--qa-config", default="configs/qa_config.yaml")
    parser.add_argument("--datagen-config", default="configs/datagen_config.yaml")
    parser.add_argument(
        "--llama-channel", choices=["openrouter", "hf_colab", "scripted"], default="scripted"
    )
    parser.add_argument(
        "--qwen-channel", choices=["openrouter", "hf_colab", "scripted"], default="scripted"
    )
    args = parser.parse_args()

    qa_config = load_config(args.qa_config)
    datagen_config = load_config(args.datagen_config)
    cross_llm_cfg = qa_config["cross_llm_audit"]

    samples = read_samples_jsonl(args.input)
    llama_client = _build_reviewer_client("llama", args.llama_channel, datagen_config)
    qwen_client = _build_reviewer_client("qwen", args.qwen_channel, datagen_config)

    reviews = run_cross_llm_audit(
        samples,
        llama_client=llama_client,
        qwen_client=qwen_client,
        fraction=cross_llm_cfg["audit_fraction"],
        seed=cross_llm_cfg["seed"],
        llama_display_name=datagen_config["models"]["llama"]["display_name"],
        qwen_display_name=datagen_config["models"]["qwen"]["display_name"],
    )
    write_reviews_jsonl(reviews, args.output)

    logger.info(f"Cross-LLM audit hoàn tất: {len(reviews)} mẫu được review, ghi tại {args.output}")


if __name__ == "__main__":
    main()
