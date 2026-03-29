"""
Data augmentation for Vietnamese emotion classification.
Applies Random Deletion, Random Swap, Random Insertion, and Back-Translation to minority classes.

Input: processed CSV with 'text' (pyvi-segmented) and 'label' (int) columns.
Output: augmented CSV with original + synthetic samples.
"""
import random
import pandas as pd
from pathlib import Path


# Emotion label index → name (for logging)
EMOTION_NAMES = {
    0: "Enjoyment", 1: "Sadness", 2: "Anger",
    3: "Fear", 4: "Disgust", 5: "Surprise", 6: "Other"
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

    def __init__(self, seed=42):
        random.seed(seed)

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
        Falls back to original text on error (rate limit, network, etc.)

        Args:
            text: Input text (pyvi-segmented or raw Vietnamese)

        Returns:
            Back-translated text, or original text on failure
        """
        try:
            from deep_translator import GoogleTranslator
            en = GoogleTranslator(source='vi', target='en').translate(text)
            vi = GoogleTranslator(source='en', target='vi').translate(en)
            return vi if vi and vi.strip() else text
        except Exception:
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


def augment_dataset(
    input_csv: str,
    output_csv: str,
    target_counts: dict,
    techniques: list = ["deletion", "swap"],
    seed: int = 42
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
        techniques: List of augmentation techniques to cycle through
                    ["deletion", "swap"]
        seed: Random seed for reproducibility

    Returns:
        Augmented DataFrame
    """
    random.seed(seed)
    augmenter = VietnameseAugmenter(seed=seed)

    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} samples from {input_csv}")

    # Show current distribution
    print("\nCurrent class distribution:")
    for label_idx, count in sorted(df['label'].value_counts().items()):
        idx: int = label_idx  # type: ignore[assignment]
        name = EMOTION_NAMES.get(idx, str(idx))
        target = target_counts.get(idx, int(count))
        print(f"  {name:12s} (class {label_idx}): {count:4d} → target {target:4d}")

    augmented_rows = []

    for class_idx, target in target_counts.items():
        class_df = df[df['label'] == class_idx]
        current_count = len(class_df)
        needed = target - current_count

        if needed <= 0:
            name = EMOTION_NAMES.get(class_idx, str(class_idx))
            print(f"\n{name}: already has {current_count} >= {target}, skipping.")
            continue

        name = EMOTION_NAMES.get(class_idx, str(class_idx))
        print(f"\nAugmenting {name} (class {class_idx}): {current_count} → {target} (+{needed})")

        texts = class_df['text'].tolist()
        generated = 0
        technique_idx = 0

        while generated < needed:
            source_text = texts[generated % len(texts)]
            technique = techniques[technique_idx % len(techniques)]
            aug_text = augmenter.augment(source_text, technique=technique)

            # Allow duplicate if text is very short (≤3 tokens) — no choice
            is_short = len(source_text.split()) <= 3
            if aug_text != source_text or is_short:
                augmented_rows.append({'text': aug_text, 'label': class_idx})
                generated += 1

            technique_idx += 1

        print(f"  Generated {generated} augmented samples")

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

    print(f"\nAugmented dataset: {len(df)} → {len(result_df)} samples")
    print(f"Saved to {output_csv}")

    print("\nFinal class distribution:")
    for label_idx, count in sorted(result_df['label'].value_counts().items()):
        idx: int = label_idx  # type: ignore[assignment]
        name = EMOTION_NAMES.get(idx, str(idx))
        print(f"  {name:12s} (class {idx}): {count:4d}")

    return result_df


if __name__ == "__main__":
    # Run augmentation on the standard processed train set
    import os
    base_dir = Path(__file__).resolve().parents[2]
    augment_dataset(
        input_csv=str(base_dir / "data/processed/train.csv"),
        output_csv=str(base_dir / "data/processed/train_augmented.csv"),
        # Post-merge distribution (after ViGoEmotions): Anger~1091, Fear~818,
        # Disgust~1138, Surprise~1142, Other~1146, Enjoyment~1558, Sadness~947
        # Targets based on test results:
        #   Anger   (2): recall=0.35 → target 1800 (confused with Disgust, needs more)
        #   Fear    (3): ~818 samples, below median (~1100) → target 1400
        #   Disgust (4): recall=0.57 → target 1300 (cap to avoid overwhelming Anger)
        #   Surprise(5): recall=0.30 → target 2000 (lowest recall, needs most boost)
        target_counts={2: 1800, 3: 1400, 4: 1300, 5: 2000},
        techniques=["deletion", "swap", "insertion"]
    )
