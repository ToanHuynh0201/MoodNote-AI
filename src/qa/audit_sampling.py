"""
Rút mẫu audit thủ công (~150-200 mẫu) và export/import cho 2 người gán nhãn độc lập
(tác giả + 1 cộng tác viên), đúng cơ chế Cohen's Kappa thuyết minh mô tả.
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from ..data.synthetic.schema import SyntheticSample
from ..utils.emotion_constants import DEFAULT_EMOTION_LABELS, find_label_index_by_name
from ..utils.logger import get_logger

logger = get_logger("audit_sampling")


def draw_audit_sample(samples: list[SyntheticSample], n: int, seed: int) -> list[SyntheticSample]:
    """
    Rút ngẫu nhiên có seed (tái lập được) một mẫu con để audit thủ công.

    Args:
        samples: Pool mẫu synthetic đã qua dedup + leakage_guard
        n: Số mẫu muốn rút
        seed: Seed cho bộ sinh ngẫu nhiên

    Returns:
        Danh sách mẫu đã rút. Nếu n > len(samples), trả về toàn bộ pool kèm log warning
        thay vì lỗi.
    """
    if n > len(samples):
        logger.warning(
            f"draw_audit_sample: yêu cầu {n} mẫu nhưng pool chỉ có {len(samples)} — lấy toàn bộ."
        )
        n = len(samples)

    rng = random.Random(seed)
    return rng.sample(samples, n)


def export_for_raters(
    audit_samples: list[SyntheticSample],
    output_dir: str = "data/synthetic/qa/audit_sample",
    rater_names: tuple[str, str] = ("rater_a", "rater_b"),
) -> None:
    """
    Xuất mẫu audit thành 3 file CSV:
      - blind_pool.csv        (sample_id, text, model_generated_label) — chỉ để tham chiếu nội bộ
      - {rater}_sheet.csv x 2 (sample_id, text, label="") — để 2 người điền tay TÊN nhãn

    2 sheet của rater KHÔNG chứa nhãn LLM tự gán và KHÔNG chứa nhãn của người kia.

    Args:
        audit_samples: Mẫu đã rút bởi draw_audit_sample()
        output_dir: Thư mục ghi các file CSV
        rater_names: Tên 2 người gán nhãn (dùng làm tên file)
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    blind_pool = pd.DataFrame(
        {
            "sample_id": [s.sample_id for s in audit_samples],
            "text": [s.text for s in audit_samples],
            "model_generated_label": [s.label_name for s in audit_samples],
        }
    )
    blind_pool.to_csv(out_dir / "blind_pool.csv", index=False, encoding="utf-8")

    rater_sheet = pd.DataFrame(
        {
            "sample_id": [s.sample_id for s in audit_samples],
            "text": [s.text for s in audit_samples],
            "label": ["" for _ in audit_samples],
        }
    )
    for rater_name in rater_names:
        rater_sheet.to_csv(out_dir / f"{rater_name}_sheet.csv", index=False, encoding="utf-8")

    logger.info(
        f"export_for_raters: đã xuất {len(audit_samples)} mẫu cho {len(rater_names)} người "
        f"gán nhãn vào {out_dir}"
    )


def _read_rater_sheet(path: str) -> pd.DataFrame:
    """Đọc sheet rater — .xlsx (người điền tay trong Excel) hoặc .csv."""
    if str(path).lower().endswith((".xlsx", ".xlsm")):
        return pd.read_excel(path)
    return pd.read_csv(path, encoding="utf-8")


def import_rater_labels(
    rater_a_path: str, rater_b_path: str, emotion_labels: dict[int, str] | None = None
) -> pd.DataFrame:
    """
    Đọc lại 2 sheet đã điền tay, chuẩn hoá tên nhãn (case-insensitive) về số, và merge
    theo sample_id để tính Cohen's Kappa.

    Args:
        rater_a_path: Đường dẫn sheet đã điền của người thứ nhất (.csv hoặc .xlsx)
        rater_b_path: Đường dẫn sheet đã điền của người thứ hai (.csv hoặc .xlsx)
        emotion_labels: Mapping nhãn tùy chỉnh (mặc định DEFAULT_EMOTION_LABELS)

    Returns:
        DataFrame[sample_id, text, rater_a_label, rater_b_label] (nhãn dạng số).
        Cũng ghi merged_labels.csv vào cùng thư mục với rater_a_path.

    Raises:
        ValueError: nếu có tên nhãn không khớp 7 nhãn đã biết, hoặc 2 sheet lệch tập
            sample_id (lệch tập sẽ làm sai lệch Kappa một cách âm thầm nếu không chặn)
    """
    labels = emotion_labels or DEFAULT_EMOTION_LABELS

    df_a = _read_rater_sheet(rater_a_path)
    df_b = _read_rater_sheet(rater_b_path)

    if set(df_a["sample_id"]) != set(df_b["sample_id"]):
        raise ValueError(
            "import_rater_labels: 2 sheet lệch tập sample_id — không thể ghép cặp để tính Kappa."
        )

    def _to_label_index(name: str) -> int:
        idx = find_label_index_by_name(labels, str(name))
        if idx is None:
            raise ValueError(f"import_rater_labels: tên nhãn không hợp lệ: {name!r}")
        return idx

    df_a = df_a.assign(rater_a_label=df_a["label"].map(_to_label_index))
    df_b = df_b.assign(rater_b_label=df_b["label"].map(_to_label_index))

    merged = df_a[["sample_id", "text", "rater_a_label"]].merge(
        df_b[["sample_id", "rater_b_label"]], on="sample_id", how="inner"
    )

    merged_path = Path(rater_a_path).parent / "merged_labels.csv"
    merged.to_csv(merged_path, index=False, encoding="utf-8")
    logger.info(f"import_rater_labels: đã ghép {len(merged)} mẫu, ghi ra {merged_path}")

    return merged
