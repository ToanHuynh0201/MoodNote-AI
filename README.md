# MoodNote AI - Vietnamese Emotion Classification

Repo đang được xây lại từ đầu để bám sát Nội dung 1 & 2 của đề cương NCKH 2026
("Nghiên cứu phân tích cảm xúc tiếng Việt và thuật toán gợi ý âm nhạc theo cảm
xúc, nhận biết ngữ cảnh"): sinh dữ liệu nhật ký giả lập bằng LLM mã nguồn mở
(Human-in-the-loop QA) + huấn luyện/đánh giá ablation mô hình PhoBERT trên
UIT-VSMEC.

Toàn bộ code cũ (pipeline merge ViGoEmotions, model checkpoint v1) đã được gỡ
bỏ khỏi repo và lưu trữ ngoài working tree. Việc rebuild chia thành 7 phase
tuần tự; dưới đây là trạng thái hiện tại.

## Đã hoàn thành

**Phase 1/7 — Scaffold dự án + Core utils:**
- `configs/*.yaml` (model/training/api) carry-forward từ code cũ, giữ nguyên
  hyperparameter đã tuning.
- `src/utils/`: `config.py`, `logger.py`, `emotion_constants.py`,
  `keyword_extractor.py`, `database.py`, `metrics.py` (carry-forward) +
  `config_schema.py` (mới — pydantic validation cho 3 file config).
- `tests/utils/` + CI cơ bản (`ruff` + `pytest`) qua GitHub Actions.

## Roadmap (chưa làm)

2. Dữ liệu thật (UIT-VSMEC): tải + tiền xử lý, tách train/validation/test.
3. Dữ liệu giả lập bằng LLM (Llama-3-8B-Instruct, Qwen3-8B) + Human-in-the-loop QA.
4. Huấn luyện PhoBERT + ablation 3 kịch bản (real-only/synthetic-only/combined).
5. Serving layer (inference/API), sửa công thức tính intensity.
6. Mở rộng testing & CI (models/training/inference thật).
7. Tài liệu cuối (methodology, kết quả) phục vụ báo cáo NCKH.

## Cài đặt & chạy test

```bash
pip install -r requirements.txt
pytest -q
```
