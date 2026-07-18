"""Tests for exact-hash and near-duplicate removal (no I/O, no network)."""

from src.data.synthetic.dedup import exact_dedup, near_dedup
from src.data.synthetic.schema import SyntheticSample, now_iso


def _make_sample(sample_id: str, text: str, label: int = 0) -> SyntheticSample:
    return SyntheticSample(
        sample_id=sample_id,
        text=text,
        label=label,
        label_name="Enjoyment",
        model="Scripted-Test",
        channel="scripted",
        axis_style="a",
        axis_length="b",
        axis_context="c",
        prompt_template_id="diary_v1",
        generation_round=1,
        generated_at=now_iso(),
    )


def test_exact_dedup_removes_case_and_whitespace_variant_duplicates():
    samples = [
        _make_sample("s1", "Hôm nay tôi rất vui."),
        _make_sample("s2", "  hôm nay   tôi rất vui.  "),
        _make_sample("s3", "Một câu hoàn toàn khác."),
    ]

    kept, n_removed = exact_dedup(samples)

    assert [s.sample_id for s in kept] == ["s1", "s3"]
    assert n_removed == 1


def test_exact_dedup_keeps_all_when_no_duplicates():
    samples = [_make_sample("s1", "câu một"), _make_sample("s2", "câu hai")]

    kept, n_removed = exact_dedup(samples)

    assert len(kept) == 2
    assert n_removed == 0


def test_near_dedup_removes_paraphrase_above_threshold():
    samples = [
        _make_sample("s1", "hôm nay tôi rất vui vì đạt điểm cao trong kỳ thi"),
        _make_sample("s2", "hôm nay tôi rất vui vì đạt điểm cao trong kỳ thi này"),
        _make_sample("s3", "trời hôm nay mưa rất to và tôi bị ướt hết"),
    ]

    kept, dropped_report = near_dedup(samples, threshold=90.0)

    assert [s.sample_id for s in kept] == ["s1", "s3"]
    assert len(dropped_report) == 1
    assert dropped_report[0]["dropped_id"] == "s2"
    assert dropped_report[0]["matched_id"] == "s1"
    assert dropped_report[0]["similarity"] >= 90.0


def test_near_dedup_keeps_dissimilar_samples():
    samples = [
        _make_sample("s1", "hôm nay tôi rất vui"),
        _make_sample("s2", "công việc hôm nay thật mệt mỏi và áp lực"),
    ]

    kept, dropped_report = near_dedup(samples, threshold=90.0)

    assert len(kept) == 2
    assert dropped_report == []
