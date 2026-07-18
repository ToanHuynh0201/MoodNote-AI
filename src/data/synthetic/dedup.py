"""
Loại bỏ trùng lặp exact-hash và near-duplicate trong pool mẫu synthetic.
"""
from __future__ import annotations

from ...utils.logger import get_logger
from .schema import SyntheticSample

logger = get_logger("dedup")


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _similarity_ratio(a: str, b: str) -> float:
    """
    Seam riêng cho việc tính độ tương đồng: dùng rapidfuzz hiện tại; có thể đổi sang
    difflib.SequenceMatcher (stdlib) bằng cách sửa duy nhất hàm này nếu rapidfuzz không
    có wheel cho phiên bản Python/CI đang dùng.
    """
    from rapidfuzz import fuzz

    return fuzz.token_sort_ratio(a, b)


def exact_dedup(samples: list[SyntheticSample]) -> tuple[list[SyntheticSample], int]:
    """
    Loại bỏ các mẫu trùng text hoàn toàn (sau chuẩn hoá), giữ bản xuất hiện đầu tiên.

    Args:
        samples: Danh sách mẫu synthetic cần lọc

    Returns:
        (kept, n_removed)
    """
    seen: set[str] = set()
    kept: list[SyntheticSample] = []
    n_removed = 0

    for sample in samples:
        key = _normalize(sample.text)
        if key in seen:
            n_removed += 1
            continue
        seen.add(key)
        kept.append(sample)

    if n_removed:
        logger.info(f"exact_dedup: loại {n_removed} mẫu trùng text hoàn toàn.")

    return kept, n_removed


def near_dedup(
    samples: list[SyntheticSample], threshold: float = 92.0
) -> tuple[list[SyntheticSample], list[dict]]:
    """
    Loại bỏ near-duplicate theo kiểu streaming tham lam: mỗi mẫu ứng viên chỉ so với
    các mẫu ĐÃ được giữ lại (không so toàn bộ O(n^2) mỗi lần) — đủ nhanh ở quy mô mục
    tiêu ~2.800 mẫu.

    Args:
        samples: Danh sách mẫu synthetic cần lọc (nên chạy sau exact_dedup)
        threshold: Ngưỡng rapidfuzz token_sort_ratio (0-100) để coi là near-duplicate

    Returns:
        (kept, dropped_report) — dropped_report: [{dropped_id, matched_id, similarity}, ...]
    """
    kept: list[SyntheticSample] = []
    dropped_report: list[dict] = []

    for sample in samples:
        matched = None
        best_similarity = 0.0

        for kept_sample in kept:
            similarity = _similarity_ratio(sample.text, kept_sample.text)
            if similarity >= threshold and similarity > best_similarity:
                matched = kept_sample
                best_similarity = similarity

        if matched is not None:
            dropped_report.append(
                {
                    "dropped_id": sample.sample_id,
                    "matched_id": matched.sample_id,
                    "similarity": best_similarity,
                }
            )
            continue

        kept.append(sample)

    if dropped_report:
        logger.info(f"near_dedup: loại {len(dropped_report)} mẫu near-duplicate (ngưỡng {threshold}).")

    return kept, dropped_report
