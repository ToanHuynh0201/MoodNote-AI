"""Tests for the YAKE-based Vietnamese keyword extractor."""

from src.utils.keyword_extractor import VietnameseKeywordExtractor

SAMPLE = "Hôm nay tôi rất buồn và mệt mỏi vì bị mất việc làm, không biết làm gì tiếp theo"


def test_extract_returns_list_of_strings():
    extractor = VietnameseKeywordExtractor()
    keywords = extractor.extract(SAMPLE, n=5)
    assert isinstance(keywords, list)
    assert all(isinstance(kw, str) for kw in keywords)


def test_extract_clamps_count_to_max():
    extractor = VietnameseKeywordExtractor(max_keywords=10)
    # Requesting more than max_keywords should never exceed the configured ceiling.
    assert len(extractor.extract(SAMPLE, n=50)) <= 10


def test_extract_empty_text_returns_empty():
    extractor = VietnameseKeywordExtractor()
    assert extractor.extract("", n=5) == []
    assert extractor.extract("   ", n=5) == []


def test_extract_strips_underscores_from_segmented_tokens():
    extractor = VietnameseKeywordExtractor()
    keywords = extractor.extract("hôm_nay tôi rất vui_vẻ và hạnh_phúc", n=3)
    assert all("_" not in kw for kw in keywords)
