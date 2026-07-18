"""
Bài kiểm thử tích hợp toàn bộ pipeline phase 3, HOÀN TOÀN OFFLINE (không gọi API/GPU
thật — dùng ScriptedLLMClient). Đây là "kiểm thử thủ công" thay thế cho việc chạy tay
pipeline thật mà phase 1/2 đã làm, vì phase này không có API/GPU thật trong phiên code.

Luồng: generate -> dedup -> leakage_guard -> audit 2 người (mù nhau) -> Cohen's Kappa ->
cross-LLM audit -> acceptance gate -> accepted/{train,validation,test}.csv
"""

import json
import re
from types import SimpleNamespace

import pandas as pd

from src.data.synthetic.dedup import exact_dedup, near_dedup
from src.data.synthetic.generate import generate_dataset
from src.data.synthetic.leakage_guard import find_synthetic_leakage
from src.data.synthetic.llm_client import ScriptedLLMClient
from src.data.synthetic.schema import read_samples_jsonl, write_samples_jsonl
from src.qa.acceptance_gate import run_acceptance_gate
from src.qa.audit_sampling import draw_audit_sample, export_for_raters, import_rater_labels
from src.qa.cross_llm_check import CrossLLMFlag, run_cross_llm_audit, write_reviews_jsonl
from src.qa.kappa import compute_cohens_kappa

_DATAGEN_CONFIG_YAML = """
seed: 1
generation:
  target_per_label: 3
  split_ratios:
    train: 0.7
    validation: 0.15
    test: 0.15
  max_new_tokens: 50
  temperature: 0.5
  top_p: 0.9
  log_every_n_samples: 100
diversity_axes:
  van_phong: ["style_a", "style_b"]
  do_dai: ["ngắn", "vừa"]
  ngu_canh: ["context_a", "context_b"]
"""

# Ngưỡng thấp/cao cố tình để đợt fixture nhỏ này chắc chắn "cần hiệu chỉnh prompt" —
# mục tiêu là kiểm tra cơ chế gắn cờ hoạt động, không phải mô phỏng 1 đợt sinh thật đạt chuẩn.
_QA_CONFIG_YAML = """
manual_audit:
  min_cohens_kappa: 0.9
cross_llm_audit:
  max_unnatural_rate: 0.01
  max_label_mismatch_rate: 0.01
acceptance:
  drop_cross_llm_flagged: true
"""

# 21 câu khác chủ đề rõ rệt (7 nhãn x 3 câu/nhãn) — tránh near-dup giả do câu quá giống nhau.
_CANNED_DIARY_SENTENCES = [
    "Hôm nay tôi đi học và gặp lại người bạn cũ sau nhiều năm.",
    "Buổi chiều nay trời mưa rất to khiến tôi bị kẹt xe hơn một giờ.",
    "Tôi vừa hoàn thành một dự án lớn ở công ty và cảm thấy nhẹ nhõm.",
    "Mẹ tôi nấu món canh chua yêu thích khiến cả nhà rất vui.",
    "Con mèo của tôi bị ốm nên tôi phải đưa nó đi khám thú y.",
    "Tôi xem một bộ phim kinh dị tối qua và giờ vẫn còn thấy sợ.",
    "Kỳ thi cuối kỳ sắp tới khiến tôi cảm thấy áp lực nặng nề.",
    "Tôi vừa nhận được tin nhắn bất ngờ từ một người bạn thân.",
    "Chiếc xe đạp của tôi bị hỏng phanh giữa đường về nhà.",
    "Tôi đã tiết kiệm đủ tiền để mua chiếc laptop mới mong muốn.",
    "Buổi họp nhóm hôm nay kéo dài hơn dự kiến và khá căng thẳng.",
    "Tôi trồng một chậu hoa nhỏ trên ban công và nó vừa nở.",
    "Người yêu cũ của tôi bất ngờ nhắn tin hỏi thăm sức khỏe.",
    "Tôi bị lạc đường khi đi du lịch một mình ở thành phố mới.",
    "Sáng nay tôi dậy sớm để chạy bộ quanh công viên gần nhà.",
    "Tôi vừa cãi nhau với em trai vì một chuyện rất nhỏ nhặt.",
    "Công việc part-time mới giúp tôi có thêm thu nhập mỗi tháng.",
    "Tôi lỡ làm vỡ chiếc cốc kỷ niệm mà bà ngoại tặng.",
    "Cả lớp tôi tổ chức sinh nhật bất ngờ cho một bạn cùng lớp.",
    "Tôi ngồi một mình ngắm hoàng hôn và suy nghĩ vẩn vơ.",
    "Chuyến xe buýt bị trễ khiến tôi suýt lỡ cuộc phỏng vấn quan trọng.",
]


def test_full_synthetic_pipeline_offline_smoke(tmp_path):
    datagen_config_path = tmp_path / "datagen_config.yaml"
    datagen_config_path.write_text(_DATAGEN_CONFIG_YAML, encoding="utf-8")

    # 1. Sinh dữ liệu (7 nhãn x 3 mẫu = 21 mẫu), hoàn toàn offline qua ScriptedLLMClient.
    client = ScriptedLLMClient(responses=_CANNED_DIARY_SENTENCES)
    raw_path = tmp_path / "raw.jsonl"
    generate_dataset(
        client=client,
        model_display_name="Llama-3-8B-Instruct",
        channel="scripted",
        output_path=str(raw_path),
        generation_round=1,
        config_path=str(datagen_config_path),
    )
    samples = read_samples_jsonl(raw_path)
    assert len(samples) == 21

    # 2. Dedup — 21 câu khác chủ đề nhau, không có duplicate thật.
    after_exact, n_exact_removed = exact_dedup(samples)
    after_near, near_report = near_dedup(after_exact, threshold=97.0)
    assert n_exact_removed == 0
    assert near_report == []
    assert len(after_near) == 21

    # 3. Leakage guard — 1 dòng VSMEC giả cố tình trùng với 1 mẫu vừa sinh.
    leaked_text = _CANNED_DIARY_SENTENCES[4]
    real_texts = {"train": [], "validation": [], "test": [leaked_text]}
    hits = find_synthetic_leakage(after_near, real_texts, near_dup_threshold=90.0)
    assert len(hits) == 1

    clean = [s for s in after_near if s.sample_id not in hits]
    assert len(clean) == 20

    # 4. Audit 2 người mù nhau — có 1 điểm bất đồng cố tình để Kappa không suy biến.
    audit_dir = tmp_path / "audit_sample"
    audit_sample = draw_audit_sample(clean, n=len(clean), seed=1)
    export_for_raters(audit_sample, output_dir=str(audit_dir))

    rater_a_path = audit_dir / "rater_a_sheet.csv"
    rater_b_path = audit_dir / "rater_b_sheet.csv"
    rater_a_df = pd.read_csv(rater_a_path)
    rater_b_df = pd.read_csv(rater_b_path)

    label_by_id = {s.sample_id: s.label_name for s in audit_sample}
    rater_a_df["label"] = rater_a_df["sample_id"].map(label_by_id)
    rater_b_df["label"] = rater_b_df["sample_id"].map(label_by_id)

    disagreement_id = rater_b_df["sample_id"].iloc[0]
    true_label = label_by_id[disagreement_id]
    other_label = next(name for name in label_by_id.values() if name != true_label)
    rater_b_df.loc[rater_b_df["sample_id"] == disagreement_id, "label"] = other_label

    rater_a_df.to_csv(rater_a_path, index=False, encoding="utf-8")
    rater_b_df.to_csv(rater_b_path, index=False, encoding="utf-8")

    merged = import_rater_labels(str(rater_a_path), str(rater_b_path))
    kappa = compute_cohens_kappa(merged["rater_a_label"].tolist(), merged["rater_b_label"].tolist())
    assert kappa is not None
    assert -1.0 <= kappa <= 1.0

    kappa_report_path = audit_dir / "kappa_report.json"
    kappa_report_path.write_text(
        json.dumps({"n_samples": len(merged), "kappa": kappa, "threshold": 0.9, "passed": False}),
        encoding="utf-8",
    )

    # 5. Cross-LLM audit — mọi mẫu ở đây do "Llama" sinh nên được định tuyến review bởi
    # "Qwen". Reviewer luôn echo đúng nhãn thật (tránh LABEL_MISMATCH giả do canned response
    # cố định), nhưng cố tình trả lời sai định dạng ở lần gọi đầu để đảm bảo có ít nhất 1 mẫu
    # bị gắn cờ UNNATURAL_STYLE — mà không làm rớt cả pool xuống quá ít mẫu để chia 70/15/15.
    class _ControlledReviewerClient:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, prompt, max_tokens=60, temperature=0.0, top_p=1.0):
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(text="câu trả lời sai định dạng")
            label_name = re.search(r"Nhãn cảm xúc được gán trước: (.+)\.", prompt).group(1)
            return SimpleNamespace(text=f"NHÃN: {label_name}\nTỰ_NHIÊN: có")

    llama_reviewer = _ControlledReviewerClient()  # không có mẫu Qwen nào ở test này
    qwen_reviewer = _ControlledReviewerClient()

    reviews = run_cross_llm_audit(
        clean, llama_client=llama_reviewer, qwen_client=qwen_reviewer, fraction=1.0, seed=1
    )
    assert len(reviews) == 20
    assert sum(1 for r in reviews if r.flag != CrossLLMFlag.OK) == 1
    assert any(r.flag == CrossLLMFlag.UNNATURAL_STYLE for r in reviews)

    reviews_path = tmp_path / "cross_llm_review.jsonl"
    write_reviews_jsonl(reviews, reviews_path)
    clean_path = tmp_path / "clean.jsonl"
    write_samples_jsonl(clean, clean_path)

    qa_config_path = tmp_path / "qa_config.yaml"
    qa_config_path.write_text(_QA_CONFIG_YAML, encoding="utf-8")

    # 6. Acceptance gate — luôn ghi accepted/*.csv, loại đúng mẫu bị gắn cờ, và verdict
    # phải báo "cần hiệu chỉnh prompt" vì fixture cố tình cho ngưỡng thấp/tỉ lệ lỗi cao.
    output_dir = tmp_path / "accepted"
    verdict = run_acceptance_gate(
        clean_samples_path=str(clean_path),
        cross_llm_reviews_path=str(reviews_path),
        kappa_report_path=str(kappa_report_path),
        qa_config_path=str(qa_config_path),
        datagen_config_path=str(datagen_config_path),
        output_dir=str(output_dir),
        generation_round=1,
        strict=False,
    )

    assert verdict.needs_prompt_revision is True

    train_df = pd.read_csv(output_dir / "train.csv")
    val_df = pd.read_csv(output_dir / "validation.csv")
    test_df = pd.read_csv(output_dir / "test.csv")
    assert list(train_df.columns) == ["sample_id", "text", "label", "model", "generation_round"]

    all_ids = set(train_df["sample_id"]) | set(val_df["sample_id"]) | set(test_df["sample_id"])
    flagged_ids = {r.sample_id for r in reviews if r.flag != CrossLLMFlag.OK}

    assert not (flagged_ids & all_ids)
    assert len(all_ids) == 20 - len(flagged_ids)
