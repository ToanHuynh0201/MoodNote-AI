"""
Data augmentation for Vietnamese emotion classification.
Applies Random Deletion, Random Swap, Random Insertion, and Back-Translation to minority classes.

Input: processed CSV with 'text' (pyvi-segmented) and 'label' (int) columns.
Output: augmented CSV with original + synthetic samples.
"""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from ..utils.emotion_constants import DEFAULT_EMOTION_LABELS
from ..utils.logger import get_logger

logger = get_logger("augment")

# Emotion label index → name (for logging); single source of truth in emotion_constants.
EMOTION_NAMES = DEFAULT_EMOTION_LABELS

# Default augmentation plan (shared by scripts/prepare_data.py and scripts/augment_colab.py).
# Post-merge distribution (after ViGoEmotions): Anger~1091, Fear~818, Disgust~1138,
# Surprise~1142, Other~1146, Enjoyment~1558, Sadness~947.
DEFAULT_AUGMENT_TARGETS: dict[int, int] = {0: 2000, 2: 1800, 3: 1200, 4: 1100, 5: 1800}
DEFAULT_TECHNIQUES: list[str] = ["deletion", "swap"]
# Classes 0/2/5 benefit from back-translation (semantically diverse paraphrases).
DEFAULT_CLASS_TECHNIQUES: dict[int, list[str]] = {
    0: ["back_translation", "swap"],
    2: ["back_translation", "swap"],
    5: ["back_translation", "swap"],
}


class VietnameseAugmenter:
    """
    Text augmentation for pyvi-segmented Vietnamese text.

    Techniques:
    - random_deletion: Remove tokens with probability p
    - random_swap: Swap two random tokens n times
    - random_insertion: Insert a copy of an existing token at a random position
    - back_translation: Vietnamese → English → Vietnamese (requires deep_translator)
    """

    def __init__(self, seed: int = 42) -> None:
        random.seed(seed)
        self._bt_import_warned = False
        self._bt_error_warned = False

    def random_deletion(self, text: str, p: float = 0.20) -> str:
        """
        Randomly delete tokens from segmented text.

        Args:
            text: pyvi-segmented text (tokens separated by spaces)
            p: probability of deleting each token

        Returns:
            Augmented text (at least 1 token preserved)
        """
        tokens = text.split()
        if len(tokens) == 1:
            return text

        kept = [tok for tok in tokens if random.random() > p]

        # Always keep at least 1 token
        if not kept:
            kept = [random.choice(tokens)]

        return " ".join(kept)

    def random_swap(self, text: str, n: int = 2) -> str:
        """
        Randomly swap two tokens in the text, n times.

        Args:
            text: pyvi-segmented text
            n: number of swap operations

        Returns:
            Augmented text
        """
        tokens = text.split()
        if len(tokens) < 2:
            return text

        tokens = tokens.copy()
        for _ in range(n):
            idx1, idx2 = random.sample(range(len(tokens)), 2)
            tokens[idx1], tokens[idx2] = tokens[idx2], tokens[idx1]

        return " ".join(tokens)

    def random_insertion(self, text: str, n: int = 1) -> str:
        """
        Randomly insert a copy of an existing token at a random position, n times.

        Picks a word already present in the sentence to preserve in-domain
        vocabulary and semantic meaning. No external dictionary needed.

        Args:
            text: pyvi-segmented text
            n: number of insertion operations

        Returns:
            Augmented text
        """
        tokens = text.split()
        if len(tokens) == 0:
            return text

        tokens = tokens.copy()
        for _ in range(n):
            insert_word = random.choice(tokens)
            insert_pos = random.randint(0, len(tokens))
            tokens.insert(insert_pos, insert_word)

        return " ".join(tokens)

    def back_translate(self, text: str) -> str:
        """
        Back-translate: Vietnamese → English → Vietnamese.

        Creates semantically diverse paraphrases that differ in word choice
        while preserving the original meaning. Much more effective than
        random deletion/swap for learning diverse representations.

        Requires: pip install deep_translator
        Returns original text on failure (rate limit, network, etc.) and
        prints a warning on the first failure encountered.

        Args:
            text: Input text (pyvi-segmented or raw Vietnamese)

        Returns:
            Back-translated text, or original text on failure
        """
        try:
            from deep_translator import GoogleTranslator
            en = GoogleTranslator(source='vi', target='en').translate(text)
            if not en or not en.strip():
                return text
            vi = GoogleTranslator(source='en', target='vi').translate(en)
            return vi if vi and vi.strip() else text
        except ImportError:
            if not self._bt_import_warned:
                logger.warning(
                    "[back_translate] deep_translator not installed. "
                    "Run: pip install deep_translator"
                )
                self._bt_import_warned = True
            return text
        except Exception as e:
            if not self._bt_error_warned:
                logger.warning(f"[back_translate] Error: {e}")
                self._bt_error_warned = True
            return text

    def augment(self, text: str, technique: str = "deletion") -> str:
        """
        Apply a single augmentation technique.

        Args:
            text: Input text
            technique: "deletion", "swap", "insertion", or "back_translation"

        Returns:
            Augmented text
        """
        if technique == "deletion":
            return self.random_deletion(text)
        elif technique == "swap":
            return self.random_swap(text)
        elif technique == "insertion":
            return self.random_insertion(text)
        elif technique == "back_translation":
            return self.back_translate(text)
        else:
            raise ValueError(
                f"Unknown technique: {technique}. "
                f"Use 'deletion', 'swap', 'insertion', or 'back_translation'."
            )


def _log_distribution(df: pd.DataFrame, header: str, targets: dict[int, int] | None = None) -> None:
    """Log per-class sample counts, optionally annotated with target counts."""
    logger.info(header)
    for label_idx, count in sorted(df['label'].value_counts().items()):
        idx = int(label_idx)
        name = EMOTION_NAMES.get(idx, str(idx))
        if targets is not None:
            target = targets.get(idx, int(count))
            logger.info(f"  {name:12s} (class {idx}): {count:4d} → target {target:4d}")
        else:
            logger.info(f"  {name:12s} (class {idx}): {count:4d}")


def augment_dataset(
    input_csv: str,
    output_csv: str,
    target_counts: dict[int, int],
    techniques: list[str] | None = None,
    class_techniques: dict[int, list[str]] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Augment minority classes in a processed dataset to reach target counts.

    Args:
        input_csv: Path to processed CSV (columns: 'text', 'label')
        output_csv: Path to save augmented CSV
        target_counts: Dict mapping class_idx (int) → target sample count
                       e.g. {2: 700, 3: 700, 5: 600}
                       Only augments classes with fewer samples than target.
                       Classes NOT in this dict are left unchanged.
        techniques: Default list of augmentation techniques to cycle through
                    (defaults to DEFAULT_TECHNIQUES).
        class_techniques: Optional per-class technique override.
                          e.g. {2: ["back_translation", "swap"], 5: ["back_translation"]}
                          Classes not listed here fall back to `techniques`.
        seed: Random seed for reproducibility

    Returns:
        Augmented DataFrame
    """
    techniques = techniques or DEFAULT_TECHNIQUES
    class_techniques = class_techniques or {}

    random.seed(seed)
    augmenter = VietnameseAugmenter(seed=seed)

    df = pd.read_csv(input_csv)
    logger.info(f"Loaded {len(df)} samples from {input_csv}")
    _log_distribution(df, "Current class distribution:", targets=target_counts)

    augmented_rows = []

    for class_idx, target in target_counts.items():
        class_df = df[df['label'] == class_idx]
        current_count = len(class_df)
        needed = target - current_count
        name = EMOTION_NAMES.get(class_idx, str(class_idx))

        if needed <= 0:
            logger.info(f"{name}: already has {current_count} >= {target}, skipping.")
            continue

        logger.info(f"Augmenting {name} (class {class_idx}): {current_count} → {target} (+{needed})")

        texts = class_df['text'].tolist()
        active_techniques = class_techniques.get(class_idx, techniques)
        logger.info(f"  Techniques: {active_techniques}")
        generated = 0
        technique_idx = 0

        while generated < needed:
            source_text = texts[generated % len(texts)]
            technique = active_techniques[technique_idx % len(active_techniques)]
            aug_text = augmenter.augment(source_text, technique=technique)

            # Allow duplicate if text is very short (≤3 tokens) — no choice
            is_short = len(source_text.split()) <= 3
            if aug_text != source_text or is_short:
                augmented_rows.append({'text': aug_text, 'label': class_idx})
                generated += 1

            technique_idx += 1

        logger.info(f"  Generated {generated} augmented samples")

    if augmented_rows:
        aug_df = pd.DataFrame(augmented_rows)
        result_df = pd.concat([df, aug_df], ignore_index=True)
        # Shuffle to mix original and augmented
        result_df = result_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    else:
        result_df = df.copy()

    # Save
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_csv, index=False, encoding='utf-8')

    logger.info(f"Augmented dataset: {len(df)} → {len(result_df)} samples")
    logger.info(f"Saved to {output_csv}")
    _log_distribution(result_df, "Final class distribution:")

    return result_df


if __name__ == "__main__":
    # Run augmentation on the standard processed train set.
    base_dir = Path(__file__).resolve().parents[2]
    augment_dataset(
        input_csv=str(base_dir / "data/processed/train.csv"),
        output_csv=str(base_dir / "data/processed/train_augmented.csv"),
        target_counts=DEFAULT_AUGMENT_TARGETS,
        class_techniques=DEFAULT_CLASS_TECHNIQUES,
        techniques=["swap", "insertion"],
    )
