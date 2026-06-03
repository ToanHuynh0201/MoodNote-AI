"""
Vietnamese keyword extraction using YAKE algorithm
"""

from __future__ import annotations


class VietnameseKeywordExtractor:
    """
    Keyword extractor for Vietnamese text using YAKE (Yet Another Keyword Extractor).

    YAKE is an unsupervised, language-independent keyword extractor that works
    well with Vietnamese text pre-segmented by pyvi (words joined by underscores).
    Lower YAKE score = more important keyword.
    """

    def __init__(self, max_keywords: int = 10) -> None:
        """
        Initialize YAKE keyword extractor.

        Args:
            max_keywords: Maximum number of keywords to extract (upper bound)
        """
        try:
            import yake
        except ImportError as exc:
            raise ImportError(
                "YAKE is required for keyword extraction. Install it with: pip install yake"
            ) from exc

        self.max_keywords = max_keywords
        self.extractor = yake.KeywordExtractor(
            lan="vi",
            n=1,  # unigrams only
            dedupLim=0.7,  # deduplication threshold
            top=max_keywords,
            features=None,
        )

    def extract(self, text: str, n: int = 5) -> list[str]:
        """
        Extract keywords from Vietnamese text.

        Args:
            text: Vietnamese text (pyvi-segmented text works best)
            n: Number of keywords to return (clamped to 3-10)

        Returns:
            List[str]: Extracted keywords sorted by importance
        """
        n = max(3, min(n, self.max_keywords))

        if not text or not text.strip():
            return []

        keywords = self.extractor.extract_keywords(text)

        # YAKE returns (keyword, score) — lower score = more important
        return [kw[0].replace("_", " ") for kw in keywords[:n]]
