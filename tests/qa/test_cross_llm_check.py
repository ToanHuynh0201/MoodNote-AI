"""Tests for cross-LLM audit routing/parsing (network-free: fake reviewer clients)."""

from types import SimpleNamespace

from src.data.synthetic.schema import SyntheticSample, now_iso
from src.qa.cross_llm_check import (
    CrossLLMFlag,
    parse_review_response,
    run_cross_llm_audit,
    select_cross_llm_audit_pool,
)


class _FakeReviewerClient:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.calls: list[str] = []

    def generate(self, prompt, max_tokens=60, temperature=0.0, top_p=1.0):
        self.calls.append(prompt)
        return SimpleNamespace(text=self.response_text)


class _RaisingReviewerClient:
    def generate(self, prompt, max_tokens=60, temperature=0.0, top_p=1.0):
        raise RuntimeError("simulated reviewer failure")


def _make_sample(sample_id: str, model: str, label: int = 0) -> SyntheticSample:
    return SyntheticSample(
        sample_id=sample_id,
        text="mẫu nhật ký test",
        label=label,
        label_name="Enjoyment" if label == 0 else "Sadness",
        model=model,
        channel="scripted",
        axis_style="a",
        axis_length="b",
        axis_context="c",
        prompt_template_id="diary_v1",
        generation_round=1,
        generated_at=now_iso(),
    )


def _make_samples(n: int) -> list[SyntheticSample]:
    return [_make_sample(f"s{i}", model="Llama-3-8B-Instruct") for i in range(n)]


def test_select_cross_llm_audit_pool_size_matches_fraction():
    pool = select_cross_llm_audit_pool(_make_samples(20), fraction=0.25, seed=5)

    assert len(pool) == 5


def test_select_cross_llm_audit_pool_deterministic_with_seed():
    samples = _make_samples(20)

    first = select_cross_llm_audit_pool(samples, fraction=0.25, seed=5)
    second = select_cross_llm_audit_pool(samples, fraction=0.25, seed=5)

    assert [s.sample_id for s in first] == [s.sample_id for s in second]


def test_parse_review_response_extracts_label_and_ok_flag():
    label, flag = parse_review_response("NHÃN: Sadness\nTỰ_NHIÊN: có")

    assert label == 1
    assert flag == CrossLLMFlag.OK


def test_parse_review_response_unnatural_flag_when_answer_is_no():
    label, flag = parse_review_response("NHÃN: Sadness\nTỰ_NHIÊN: không")

    assert label == 1
    assert flag == CrossLLMFlag.UNNATURAL_STYLE


def test_parse_review_response_falls_back_on_malformed_reply(caplog):
    with caplog.at_level("WARNING"):
        label, flag = parse_review_response("câu trả lời không đúng định dạng gì cả")

    assert label is None
    assert flag == CrossLLMFlag.UNNATURAL_STYLE
    assert "không parse được" in caplog.text


def test_run_cross_llm_audit_routes_llama_sample_to_qwen_reviewer():
    samples = [_make_sample("s1", model="Llama-3-8B-Instruct", label=0)]
    llama_reviewer = _FakeReviewerClient("NHÃN: Enjoyment\nTỰ_NHIÊN: có")
    qwen_reviewer = _FakeReviewerClient("NHÃN: Enjoyment\nTỰ_NHIÊN: có")

    reviews = run_cross_llm_audit(
        samples, llama_client=llama_reviewer, qwen_client=qwen_reviewer, fraction=1.0, seed=1
    )

    assert len(qwen_reviewer.calls) == 1
    assert len(llama_reviewer.calls) == 0
    assert reviews[0].reviewer_model == "Qwen3-8B"
    assert reviews[0].flag == CrossLLMFlag.OK


def test_run_cross_llm_audit_routes_qwen_sample_to_llama_reviewer():
    samples = [_make_sample("s1", model="Qwen3-8B", label=0)]
    llama_reviewer = _FakeReviewerClient("NHÃN: Enjoyment\nTỰ_NHIÊN: có")
    qwen_reviewer = _FakeReviewerClient("NHÃN: Enjoyment\nTỰ_NHIÊN: có")

    reviews = run_cross_llm_audit(
        samples, llama_client=llama_reviewer, qwen_client=qwen_reviewer, fraction=1.0, seed=1
    )

    assert len(llama_reviewer.calls) == 1
    assert len(qwen_reviewer.calls) == 0
    assert reviews[0].reviewer_model == "Llama-3-8B-Instruct"


def test_run_cross_llm_audit_flags_label_mismatch():
    samples = [_make_sample("s1", model="Llama-3-8B-Instruct", label=0)]  # Enjoyment
    qwen_reviewer = _FakeReviewerClient("NHÃN: Sadness\nTỰ_NHIÊN: có")

    reviews = run_cross_llm_audit(
        samples,
        llama_client=_FakeReviewerClient(""),
        qwen_client=qwen_reviewer,
        fraction=1.0,
        seed=1,
    )

    assert reviews[0].flag == CrossLLMFlag.LABEL_MISMATCH
    assert reviews[0].reviewer_label == 1


def test_run_cross_llm_audit_skips_unrecognized_model_source(caplog):
    samples = [_make_sample("s1", model="Unknown-Model", label=0)]

    with caplog.at_level("WARNING"):
        reviews = run_cross_llm_audit(
            samples,
            llama_client=_FakeReviewerClient(""),
            qwen_client=_FakeReviewerClient(""),
            fraction=1.0,
            seed=1,
        )

    assert reviews == []
    assert "không nhận diện được model nguồn" in caplog.text


def test_run_cross_llm_audit_handles_reviewer_client_error():
    samples = [_make_sample("s1", model="Llama-3-8B-Instruct", label=0)]

    reviews = run_cross_llm_audit(
        samples,
        llama_client=_FakeReviewerClient(""),
        qwen_client=_RaisingReviewerClient(),
        fraction=1.0,
        seed=1,
    )

    assert reviews[0].flag == CrossLLMFlag.UNNATURAL_STYLE
    assert reviews[0].reviewer_label is None
