"""
Cross-LLM audit: model KHÁC với model đã sinh ra mẫu sẽ review lại một phần dữ liệu
(Qwen review mẫu do Llama sinh, và ngược lại), trên ~15-20% tổng dữ liệu.
"""

from __future__ import annotations

import random
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel

from ..data.synthetic.llm_client import BaseLLMClient
from ..data.synthetic.schema import SyntheticSample
from ..utils.emotion_constants import DEFAULT_EMOTION_LABELS, find_label_index_by_name
from ..utils.logger import get_logger

logger = get_logger("cross_llm_check")


class CrossLLMFlag(StrEnum):
    OK = "ok"
    UNNATURAL_STYLE = "unnatural_style"  # "rập khuôn / thiếu tự nhiên"
    LABEL_MISMATCH = "label_mismatch"


class CrossLLMReview(BaseModel):
    sample_id: str
    reviewer_model: str
    reviewer_label: int | None
    flag: CrossLLMFlag
    raw_response: str | None = None


def read_reviews_jsonl(path: str | Path) -> list[CrossLLMReview]:
    """Read a JSONL file of CrossLLMReview rows (one per line)."""
    with open(path, encoding="utf-8") as f:
        return [CrossLLMReview.model_validate_json(line) for line in f if line.strip()]


def write_reviews_jsonl(reviews: list[CrossLLMReview], path: str | Path) -> None:
    """Write CrossLLMReview rows to a JSONL file, one per line."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for review in reviews:
            f.write(review.model_dump_json() + "\n")


def select_cross_llm_audit_pool(
    samples: list[SyntheticSample], fraction: float, seed: int
) -> list[SyntheticSample]:
    """
    Rút ngẫu nhiên có seed một phần pool để cross-LLM audit.

    Args:
        samples: Pool mẫu synthetic (thường là pool đã qua dedup + leakage_guard)
        fraction: Tỉ lệ cần rút (vd. 0.175 cho ~17.5%)
        seed: Seed cho bộ sinh ngẫu nhiên

    Returns:
        Danh sách mẫu đã rút, kích thước = round(len(samples) * fraction)
    """
    n = max(0, min(round(len(samples) * fraction), len(samples)))
    rng = random.Random(seed)
    return rng.sample(samples, n)


def build_review_prompt(sample: SyntheticSample) -> str:
    """Ghép prompt yêu cầu reviewer LLM trả lời theo định dạng cố định, dễ parse."""
    return (
        "Bạn là người kiểm tra chất lượng dữ liệu nhật ký tiếng Việt giả lập. Đọc đoạn "
        "nhật ký sau và trả lời ĐÚNG theo định dạng, không giải thích thêm:\n\n"
        f"Đoạn nhật ký:\n{sample.text}\n\n"
        f"Nhãn cảm xúc được gán trước: {sample.label_name}.\n\n"
        # Không liệt kê sẵn 7 nhãn thì reviewer tự đặt tên nhãn tiếng Việt ("Lo âu",
        # "Hài hước") — find_label_index_by_name() không map được và mẫu bị gắn cờ oan.
        "Chỉ được chọn ĐÚNG MỘT trong 7 nhãn sau, viết y nguyên tiếng Anh: "
        f"{', '.join(DEFAULT_EMOTION_LABELS.values())}.\n\n"
        "Trả lời theo đúng 2 dòng:\n"
        "NHÃN: <một trong 7 nhãn trên>\n"
        "TỰ_NHIÊN: <có/không — đánh giá 'không' nếu câu văn dịch nghĩa đen từ "
        "tiếng Anh, dùng sai nghĩa từ, kết hợp từ không tự nhiên, hoặc sai "
        "ngữ pháp tiếng Việt>"
    )


def parse_review_response(raw_text: str) -> tuple[int | None, CrossLLMFlag]:
    """
    Parse câu trả lời dạng cố định (NHÃN: .../TỰ_NHIÊN: có|không) của reviewer LLM.

    Không parse được (thiếu dòng, tên nhãn sai) -> log warning, trả về
    (None, CrossLLMFlag.UNNATURAL_STYLE) một cách thận trọng — không bao giờ raise,
    vì 1 câu trả lời reviewer lỗi không nên làm crash cả đợt audit.

    Args:
        raw_text: Câu trả lời thô của reviewer LLM

    Returns:
        (reviewer_label, flag) — flag chỉ phản ánh phần "TỰ_NHIÊN"; việc so khớp nhãn với
        mẫu gốc (LABEL_MISMATCH) do run_cross_llm_audit() quyết định.
    """
    label_name: str | None = None
    natural: str | None = None

    for line in raw_text.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("NHÃN:"):
            label_name = stripped.split(":", 1)[1].strip()
        elif stripped.upper().startswith("TỰ_NHIÊN:"):
            natural = stripped.split(":", 1)[1].strip().lower()

    if label_name is None or natural is None:
        logger.warning(f"parse_review_response: không parse được câu trả lời: {raw_text!r}")
        return None, CrossLLMFlag.UNNATURAL_STYLE

    reviewer_label = find_label_index_by_name(DEFAULT_EMOTION_LABELS, label_name)
    if reviewer_label is None:
        logger.warning(f"parse_review_response: tên nhãn không hợp lệ: {label_name!r}")
        return None, CrossLLMFlag.UNNATURAL_STYLE

    flag = CrossLLMFlag.OK if natural.startswith("có") else CrossLLMFlag.UNNATURAL_STYLE
    return reviewer_label, flag


def run_cross_llm_audit(
    samples: list[SyntheticSample],
    llama_client: BaseLLMClient,
    qwen_client: BaseLLMClient,
    fraction: float,
    seed: int,
    llama_display_name: str = "Llama-3-8B-Instruct",
    qwen_display_name: str = "Qwen3-8B",
) -> list[CrossLLMReview]:
    """
    Chạy cross-LLM audit: định tuyến mỗi mẫu tới model KHÁC model đã sinh ra nó.

    Args:
        samples: Pool mẫu synthetic cần audit
        llama_client: Client dùng để review các mẫu do Qwen sinh
        qwen_client: Client dùng để review các mẫu do Llama sinh
        fraction: Tỉ lệ pool cần audit (vd. 0.175)
        seed: Seed cho việc rút mẫu
        llama_display_name: Tên hiển thị của model Llama (khớp SyntheticSample.model)
        qwen_display_name: Tên hiển thị của model Qwen (khớp SyntheticSample.model)

    Returns:
        Danh sách CrossLLMReview, một phần tử cho mỗi mẫu audit thành công (mẫu có
        model nguồn không nhận diện được sẽ bị bỏ qua kèm log warning).
    """
    pool = select_cross_llm_audit_pool(samples, fraction, seed)
    reviews: list[CrossLLMReview] = []

    for sample in pool:
        model_lower = sample.model.lower()
        if model_lower.startswith("llama"):
            reviewer_client, reviewer_model = qwen_client, qwen_display_name
        elif model_lower.startswith("qwen"):
            reviewer_client, reviewer_model = llama_client, llama_display_name
        else:
            logger.warning(
                f"run_cross_llm_audit: không nhận diện được model nguồn '{sample.model}' "
                f"cho mẫu {sample.sample_id}, bỏ qua."
            )
            continue

        prompt = build_review_prompt(sample)
        try:
            response = reviewer_client.generate(prompt, max_tokens=60, temperature=0.0, top_p=1.0)
        except Exception as e:
            logger.error(f"run_cross_llm_audit: reviewer lỗi cho {sample.sample_id}: {e}")
            reviews.append(
                CrossLLMReview(
                    sample_id=sample.sample_id,
                    reviewer_model=reviewer_model,
                    reviewer_label=None,
                    flag=CrossLLMFlag.UNNATURAL_STYLE,
                    raw_response=None,
                )
            )
            continue

        reviewer_label, flag = parse_review_response(response.text)
        if flag == CrossLLMFlag.OK and reviewer_label != sample.label:
            flag = CrossLLMFlag.LABEL_MISMATCH

        reviews.append(
            CrossLLMReview(
                sample_id=sample.sample_id,
                reviewer_model=reviewer_model,
                reviewer_label=reviewer_label,
                flag=flag,
                raw_response=response.text,
            )
        )

    return reviews
