"""
Sinh prompt nhật ký theo nhãn cảm xúc: 3 trục đa dạng hóa đúng thuyết minh
(văn phong, độ dài, ngữ cảnh) + instruction giọng văn nhật ký cá nhân.
Không kèm ví dụ mẫu viết tay (quyết định đã chốt — thuần prompt-engineering).
"""
from __future__ import annotations

import random

from ...utils.emotion_constants import DEFAULT_EMOTION_LABELS

PROMPT_TEMPLATE_ID = "diary_v1"

DIARY_VOICE_INSTRUCTION = (
    "Bạn đang viết một trang nhật ký cá nhân bằng tiếng Việt, ở ngôi thứ nhất "
    "(tôi/mình), kể lại một khoảnh khắc và cảm xúc thật trong ngày của bản thân. "
    "Đây KHÔNG PHẢI một bài đăng mạng xã hội — không dùng hashtag, không emoji, "
    "không kêu gọi tương tác."
)


def build_prompt(
    label: int,
    style: str,
    length: str,
    context: str,
    emotion_labels: dict[int, str] | None = None,
) -> str:
    """
    Ghép prompt sinh dữ liệu cho 1 nhãn cảm xúc.

    Args:
        label: Chỉ số nhãn cảm xúc (0-6)
        style: Giá trị rút từ trục "văn phong"
        length: Giá trị rút từ trục "độ dài"
        context: Giá trị rút từ trục "ngữ cảnh"
        emotion_labels: Mapping nhãn tùy chỉnh (mặc định DEFAULT_EMOTION_LABELS)

    Returns:
        Chuỗi prompt hoàn chỉnh gửi cho LLM

    Raises:
        ValueError: nếu label không thuộc tập nhãn đã biết
    """
    labels = emotion_labels or DEFAULT_EMOTION_LABELS
    if label not in labels:
        raise ValueError(f"Unknown label index: {label}")

    label_name = labels[label]
    return (
        f"{DIARY_VOICE_INSTRUCTION}\n\n"
        f"Cảm xúc chủ đạo cần thể hiện: {label_name}.\n"
        f"Văn phong: {style}.\n"
        f"Độ dài: {length}.\n"
        f"Ngữ cảnh/tình huống: {context}.\n\n"
        f"Chỉ viết đúng nội dung trang nhật ký, không thêm tiêu đề hay lời dẫn."
    )


def sample_axis_values(axis_pools: dict[str, list[str]], rng: random.Random) -> dict[str, str]:
    """
    Rút ngẫu nhiên 1 giá trị cho mỗi trục đa dạng hóa.

    Args:
        axis_pools: {"van_phong": [...], "do_dai": [...], "ngu_canh": [...]}
            (đọc từ configs/datagen_config.yaml's diversity_axes)
        rng: Bộ sinh số ngẫu nhiên đã seed, để tái lập được

    Returns:
        {"style": ..., "length": ..., "context": ...}
    """
    return {
        "style": rng.choice(axis_pools["van_phong"]),
        "length": rng.choice(axis_pools["do_dai"]),
        "context": rng.choice(axis_pools["ngu_canh"]),
    }
