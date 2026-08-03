"""
Sinh prompt nhật ký theo nhãn cảm xúc: 3 trục đa dạng hóa đúng thuyết minh
(văn phong, độ dài, ngữ cảnh) + instruction giọng văn nhật ký cá nhân.
Có kèm 1 ví dụ mẫu/nhãn (đảo quyết định "thuần prompt-engineering" trước đó) vì
dữ liệu round 1 (Llama3-8B/Qwen3-8B) xuất hiện lỗi dịch nghĩa đen từ tiếng Anh
("chặn tai" thay vì "bịt tai"...) — xem data/synthetic/qa/round_verdict.json
(needs_prompt_revision: true).
"""
from __future__ import annotations

import random

from ...utils.emotion_constants import DEFAULT_EMOTION_LABELS

PROMPT_TEMPLATE_ID = "diary_v1"

DIARY_VOICE_INSTRUCTION = (
    "Bạn đang viết một trang nhật ký cá nhân bằng tiếng Việt, ở ngôi thứ nhất "
    "(tôi/mình), kể lại một khoảnh khắc và cảm xúc thật trong ngày của bản thân. "
    "Đây KHÔNG PHẢI một bài đăng mạng xã hội — không dùng hashtag, không emoji, "
    "không kêu gọi tương tác. Dùng tiếng Việt tự nhiên, đúng ngữ pháp và đúng "
    "nghĩa từ; TUYỆT ĐỐI không dịch nghĩa đen thành ngữ/tục ngữ tiếng Anh sang "
    "tiếng Việt, không chọn từ sai nghĩa hoặc ghép từ không tự nhiên."
)

EXAMPLE_BY_LABEL: dict[str, str] = {
    "Enjoyment": (
        "Hôm nay đi ăn với đám bạn thân xong về nhà mà lòng vẫn còn vui phơi "
        "phới. Được cười thả ga, nói đủ thứ chuyện trên trời dưới đất, tự "
        "nhiên thấy nhẹ cả người."
    ),
    "Sadness": (
        "Cả buổi chiều cứ ngồi thẫn thờ, chẳng muốn làm gì. Nghĩ lại chuyện "
        "lúc sáng mà lòng cứ trĩu xuống, buồn không biết chia sẻ cùng ai."
    ),
    "Anger": (
        "Bực cả người vì bị hiểu lầm mà không ai chịu nghe mình giải thích. "
        "Càng nghĩ càng tức, chỉ muốn đóng sầm cửa lại rồi ngồi im một lúc "
        "cho hạ hỏa."
    ),
    "Fear": (
        "Tự nhiên thấy lo lắng không yên, cứ nghĩ đến ngày mai là tim lại "
        "đập nhanh. Sợ mình không chuẩn bị kịp, sợ mọi thứ vượt ngoài tầm "
        "kiểm soát."
    ),
    "Disgust": (
        "Nhìn cảnh đó mà thấy ghê hết cả người, chỉ muốn quay đi ngay lập "
        "tức. Không hiểu sao người ta có thể làm chuyện như vậy được."
    ),
    "Surprise": (
        "Đang ngồi yên thì nhận được tin báo mà giật cả mình, không tin nổi "
        "vào tai mình luôn. Phải đọc lại đến hai ba lần mới dám chắc là "
        "thật."
    ),
    "Other": (
        "Một ngày bình thường trôi qua, chẳng có gì đặc biệt để kể. Làm mấy "
        "việc quen thuộc rồi lên giường đi ngủ như mọi hôm."
    ),
}


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
    example = EXAMPLE_BY_LABEL.get(label_name, "")
    return (
        f"{DIARY_VOICE_INSTRUCTION}\n\n"
        f"Ví dụ giọng văn tự nhiên (chỉ tham khảo cách hành văn, KHÔNG sao "
        f"chép nội dung):\n{example}\n\n"
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
