"""
Pure detector: mẫu synthetic nào trùng/near-duplicate với text thật VSMEC (phase 2).

Giống hệt triết lý find_exact_text_leakage() của phase 2: hàm này CHỈ phát hiện và log,
không bao giờ tự sửa/xoá dòng nào. Việc lọc bỏ mẫu synthetic bị trùng (hợp lý ở đây,
khác với test set VSMEC cố định) nằm ở lớp orchestration scripts/check_leakage.py.
"""
from __future__ import annotations

from dataclasses import dataclass

from ...utils.logger import get_logger
from .dedup import _normalize, _similarity_ratio
from .schema import SyntheticSample

logger = get_logger("leakage_guard")


@dataclass
class LeakageHit:
    sample_id: str
    matched_split: str  # "train" | "validation" | "test"
    matched_text: str
    similarity: float  # 100.0 nếu trùng hoàn toàn
    match_type: str  # "exact" | "near"


def find_synthetic_leakage(
    samples: list[SyntheticSample],
    real_texts_by_split: dict[str, list[str]],
    near_dup_threshold: float = 90.0,
) -> dict[str, LeakageHit]:
    """
    Tìm mẫu synthetic trùng/near-duplicate với bất kỳ split thật nào (VSMEC).

    Args:
        samples: Danh sách mẫu synthetic cần kiểm tra (KHÔNG bị sửa đổi)
        real_texts_by_split: vd. {"train": [...], "validation": [...], "test": [...]}
        near_dup_threshold: Ngưỡng rapidfuzz token_sort_ratio để coi là near-duplicate

    Returns:
        {sample_id: LeakageHit} cho mỗi mẫu synthetic bị phát hiện trùng
    """
    normalized_real: dict[str, list[str]] = {
        split: [_normalize(text) for text in texts] for split, texts in real_texts_by_split.items()
    }

    hits: dict[str, LeakageHit] = {}

    for sample in samples:
        normalized_sample_text = _normalize(sample.text)

        for split, real_texts in real_texts_by_split.items():
            for real_text, normalized_real_text in zip(
                real_texts, normalized_real[split], strict=True
            ):
                if normalized_sample_text == normalized_real_text:
                    hits[sample.sample_id] = LeakageHit(
                        sample_id=sample.sample_id,
                        matched_split=split,
                        matched_text=real_text,
                        similarity=100.0,
                        match_type="exact",
                    )
                    break

                similarity = _similarity_ratio(sample.text, real_text)
                if similarity >= near_dup_threshold:
                    hits[sample.sample_id] = LeakageHit(
                        sample_id=sample.sample_id,
                        matched_split=split,
                        matched_text=real_text,
                        similarity=similarity,
                        match_type="near",
                    )
                    break

            if sample.sample_id in hits:
                break

    if hits:
        logger.warning(
            f"leakage_guard: phát hiện {len(hits)}/{len(samples)} mẫu synthetic trùng/"
            "near-duplicate với dữ liệu thật VSMEC."
        )
    else:
        logger.info("leakage_guard: không phát hiện rò rỉ nào với dữ liệu thật VSMEC.")

    return hits
