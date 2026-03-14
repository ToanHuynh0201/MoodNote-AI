"""
Merge UIT-VSMEC and ViGoEmotions datasets into a unified training set.

Strategy:
- Maps ViGoEmotions 27 fine-grained labels → 7 VSMEC classes via priority cascade
- Deduplicates ViGoEmotions samples against existing VSMEC sentences
- train + val: VSMEC + ViGoEmotions
- test: VSMEC only (held-out benchmark, not contaminated)

Output: data/merged/{train,validation,test}.csv
  Columns: Sentence, Emotion  (same schema as data/raw/*.csv)
  → Can be fed directly into preprocess.py
"""
import json
import re
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd

# ── Label mapping ────────────────────────────────────────────────────────────

VIGOEMOTIONS_TO_VSMEC: dict[str, str] = {
    # Enjoyment
    "joy": "Enjoyment", "amusement": "Enjoyment", "excitement": "Enjoyment",
    "love": "Enjoyment", "desire": "Enjoyment", "optimism": "Enjoyment",
    "caring": "Enjoyment", "pride": "Enjoyment", "admiration": "Enjoyment",
    "gratitude": "Enjoyment", "relief": "Enjoyment", "approval": "Enjoyment",
    # Sadness
    "sadness": "Sadness", "grief": "Sadness",
    "disappointment": "Sadness", "remorse": "Sadness",
    "disapproval": "Sadness",   # moved from Anger — closer to disappointment in Vietnamese
    # Anger — chỉ giữ intense anger, loại bỏ ambiguous labels
    "anger": "Anger",
    # Fear
    "fear": "Fear", "nervousness": "Fear",
    # Disgust
    "disgust": "Disgust", "embarrassment": "Disgust",
    # Surprise
    "surprise": "Surprise", "curiosity": "Surprise",
    # Other — cognitive/ambiguous states không map rõ sang VSMEC
    "neutral": "Other", "annoyance": "Other",
    "confusion": "Other", "realization": "Other",
}

# Conflict resolution priority (highest → lowest)
PRIORITY: list[str] = [
    "Anger", "Fear", "Disgust", "Sadness", "Enjoyment", "Surprise", "Other"
]

VSMEC_CLASSES = list(PRIORITY)  # canonical 7-class names


# ── Core helpers ─────────────────────────────────────────────────────────────

def resolve_multilabel(fine_grained_labels: list[str]) -> str | None:
    """
    Map a list of ViGoEmotions fine-grained labels to a single VSMEC emotion.

    Rules:
    1. Map each label via VIGOEMOTIONS_TO_VSMEC (unmapped labels are ignored).
    2. If only one VSMEC class results → return it directly.
    3. If multiple classes → remove 'Other' if any non-Other class is present.
    4. Apply PRIORITY cascade and return the highest-priority class.
    5. Return None if labels list is empty or all labels are unmapped.
    """
    if not fine_grained_labels:
        return None

    mapped = {VIGOEMOTIONS_TO_VSMEC[lbl] for lbl in fine_grained_labels
              if lbl in VIGOEMOTIONS_TO_VSMEC}

    if not mapped:
        return None

    if len(mapped) > 1 and "Other" in mapped:
        mapped -= {"Other"}

    for emotion in PRIORITY:
        if emotion in mapped:
            return emotion

    return None  # should never reach here


def _normalize_sentence(text: str) -> str:
    """Lowercase + collapse whitespace for deduplication comparison."""
    return re.sub(r"\s+", " ", str(text).strip().lower())


# ── I/O helpers ──────────────────────────────────────────────────────────────

def load_vsmec_split(csv_path: str | Path) -> pd.DataFrame:
    """Load a VSMEC split CSV. Returns DataFrame with columns Sentence, Emotion."""
    df = pd.read_csv(csv_path)
    if "Sentence" not in df.columns or "Emotion" not in df.columns:
        raise ValueError(
            f"Expected columns 'Sentence' and 'Emotion' in {csv_path}, "
            f"got: {list(df.columns)}"
        )
    return df[["Sentence", "Emotion"]].copy()


def load_vigoemotions_split(csv_path: str | Path) -> pd.DataFrame:
    """
    Load a normalized ViGoEmotions split CSV.
    Returns DataFrame with columns: text (str), labels (Python list).
    """
    df = pd.read_csv(csv_path)

    if "labels" not in df.columns:
        raise ValueError(
            f"Expected column 'labels' in {csv_path}, got: {list(df.columns)}"
        )

    # Parse JSON string → Python list
    df["labels"] = df["labels"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else (x if isinstance(x, list) else [])
    )
    return df[["text", "labels"]].copy()


# ── Conversion & deduplication ───────────────────────────────────────────────

def convert_vigoemotions_to_vsmec_format(
    df: pd.DataFrame,
    single_label_only: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Apply resolve_multilabel() to every ViGoEmotions row.

    Args:
        df: ViGoEmotions DataFrame with 'text' and 'labels' columns
        single_label_only: If True, skip samples with more than 1 original label.
                           Eliminates priority-cascade noise from multi-label samples.

    Returns:
        (converted_df, stats)

        converted_df: DataFrame with columns Sentence, Emotion
                      (only rows with a clear VSMEC mapping)
        stats: {
            "total_input": int,
            "resolved": int,
            "skipped_empty": int,
            "label_distribution": Counter,
            "conflict_examples": list[dict]   # first 10 skipped rows
        }
    """
    resolved_rows = []
    skipped_empty = 0
    conflict_examples: list[dict] = []
    label_distribution: Counter = Counter()

    for _, row in df.iterrows():
        labels: list[str] = row["labels"]

        if single_label_only and len(labels) != 1:
            skipped_empty += 1
            if len(conflict_examples) < 10:
                conflict_examples.append({"text": row["text"], "labels": labels})
            continue

        vsmec_class = resolve_multilabel(labels)

        if vsmec_class is None:
            skipped_empty += 1
            if len(conflict_examples) < 10:
                conflict_examples.append({"text": row["text"], "labels": labels})
            continue

        resolved_rows.append({"Sentence": row["text"], "Emotion": vsmec_class})
        label_distribution[vsmec_class] += 1

    converted_df = pd.DataFrame(resolved_rows, columns=["Sentence", "Emotion"])

    stats = {
        "total_input": len(df),
        "resolved": len(converted_df),
        "skipped_empty": skipped_empty,
        "label_distribution": label_distribution,
        "conflict_examples": conflict_examples,
    }
    return converted_df, stats


def deduplicate(
    df: pd.DataFrame, existing_sentences: set[str]
) -> tuple[pd.DataFrame, int]:
    """
    Remove rows from df where the sentence already exists in existing_sentences.

    Comparison is done on normalized (lowercase, whitespace-collapsed) text.

    Args:
        df: DataFrame with a 'Sentence' column
        existing_sentences: set of already-normalized VSMEC sentences

    Returns:
        (deduplicated_df, n_removed)
    """
    mask = df["Sentence"].apply(_normalize_sentence).isin(existing_sentences)
    n_removed = int(mask.sum())
    return df[~mask].reset_index(drop=True), n_removed


# ── Merge orchestration ──────────────────────────────────────────────────────

def merge_split(
    vsmec_path: str | Path,
    vigoemotions_path: str | Path,
    output_path: str | Path,
    include_vigoemotions: bool = True,
    single_label_only: bool = True,
    seed: int = 42,
) -> dict:
    """
    Merge one dataset split and save to output_path.

    Steps:
      1. Load VSMEC split
      2. Load ViGoEmotions split (if include_vigoemotions)
      3. Convert ViGoEmotions → VSMEC format
      4. Deduplicate ViGoEmotions against VSMEC
      5. Concatenate VSMEC + converted ViGoEmotions
      6. Shuffle with seed
      7. Save as CSV (columns: Sentence, Emotion)

    Returns:
        stats dict with merge details
    """
    vsmec_df = load_vsmec_split(vsmec_path)
    stats = {
        "vsmec_count": len(vsmec_df),
        "vigo_total_input": 0,
        "vigo_resolved": 0,
        "vigo_skipped_empty": 0,
        "vigo_dedup_removed": 0,
        "vigo_added": 0,
        "label_distribution_vigo": Counter(),
        "conflict_examples": [],
    }

    if include_vigoemotions:
        vigo_df = load_vigoemotions_split(vigoemotions_path)
        converted_df, conv_stats = convert_vigoemotions_to_vsmec_format(vigo_df, single_label_only=single_label_only)

        stats["vigo_total_input"] = conv_stats["total_input"]
        stats["vigo_resolved"] = conv_stats["resolved"]
        stats["vigo_skipped_empty"] = conv_stats["skipped_empty"]
        stats["label_distribution_vigo"] = conv_stats["label_distribution"]
        stats["conflict_examples"] = conv_stats["conflict_examples"]

        # Deduplicate against VSMEC
        existing = {_normalize_sentence(s) for s in vsmec_df["Sentence"]}
        deduped_df, n_removed = deduplicate(converted_df, existing)
        stats["vigo_dedup_removed"] = n_removed
        stats["vigo_added"] = len(deduped_df)

        merged_df = pd.concat([vsmec_df, deduped_df], ignore_index=True)
        merged_df = merged_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    else:
        merged_df = vsmec_df.copy()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False, encoding="utf-8")

    stats["final_count"] = len(merged_df)
    stats["final_distribution"] = Counter(merged_df["Emotion"].tolist())
    return stats


# ── Reporting ────────────────────────────────────────────────────────────────

def print_merge_report(split_name: str, stats: dict) -> None:
    """Print a formatted merge report for one split."""
    print(f"\n{'='*60}")
    print(f"  {split_name.upper()} SPLIT MERGE REPORT")
    print(f"{'='*60}")
    print(f"  VSMEC samples:            {stats['vsmec_count']:6d}")

    if stats["vigo_total_input"] > 0:
        skip_pct = stats["vigo_skipped_empty"] / stats["vigo_total_input"] * 100
        dedup_pct = stats["vigo_dedup_removed"] / max(stats["vigo_resolved"], 1) * 100
        print(f"  ViGoEmotions input:       {stats['vigo_total_input']:6d}")
        print(f"  ViGoEmotions resolved:    {stats['vigo_resolved']:6d}")
        print(f"  ViGoEmotions skipped:     {stats['vigo_skipped_empty']:6d}  ({skip_pct:.1f}%)")
        print(f"  ViGoEmotions deduped:     {stats['vigo_dedup_removed']:6d}  ({dedup_pct:.1f}% of resolved)")
        print(f"  ViGoEmotions added:       {stats['vigo_added']:6d}")

        if stats["conflict_examples"]:
            print(f"\n  Skipped sample examples (first {len(stats['conflict_examples'])}):")
            for ex in stats["conflict_examples"][:5]:
                print(f"    labels={ex['labels']}  text={str(ex['text'])[:60]}")

    print(f"\n  Final count:              {stats['final_count']:6d}")
    print(f"\n  Final distribution:")
    dist = stats["final_distribution"]
    total = sum(dist.values())
    imb_ratio = max(dist.values()) / max(min(dist.values()), 1) if dist else 0.0
    for cls in VSMEC_CLASSES:
        cnt = dist.get(cls, 0)
        pct = cnt / total * 100 if total else 0
        bar = "#" * int(pct / 2)
        print(f"    {cls:12s}: {cnt:5d} ({pct:5.1f}%)  {bar}")
    print(f"\n  Imbalance ratio (max/min): {imb_ratio:.1f}x")


def _suggest_augment_targets(train_stats: dict) -> None:
    """Print suggested augment.py target_counts based on merged train distribution."""
    dist = train_stats["final_distribution"]
    if not dist:
        return

    median_count = sorted(dist.values())[len(dist) // 2]
    suggestions = {}
    for cls_name, count in dist.items():
        if count < median_count * 0.6:  # more than 40% below median
            target = int(median_count * 0.8)
            cls_idx = VSMEC_CLASSES.index(cls_name)
            suggestions[cls_idx] = target

    if suggestions:
        print(f"\n  Suggested augment.py target_counts (classes below 60% of median):")
        print(f"  target_counts = {suggestions}")
        print(f"  (Edit src/data/augment.py __main__ to use these values)")
    else:
        print(f"\n  Class distribution is balanced — augmentation may not be needed.")


# ── Main ─────────────────────────────────────────────────────────────────────

def main(
    vsmec_dir: str = "data/raw",
    vigoemotions_dir: str = "data/raw/vigoemotions",
    output_dir: str = "data/merged",
) -> None:
    """
    Merge all splits:
      - train:      VSMEC train + ViGoEmotions train
      - validation: VSMEC val   + ViGoEmotions val
      - test:       VSMEC test only (copy as-is, benchmark stays clean)
    """
    vsmec_path = Path(vsmec_dir)
    vigo_path = Path(vigoemotions_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("Starting dataset merge: UIT-VSMEC + ViGoEmotions")
    print(f"  VSMEC source:       {vsmec_path}")
    print(f"  ViGoEmotions source:{vigo_path}")
    print(f"  Output:             {out_path}")

    all_stats = {}

    # train + validation: merge both sources
    for split in ["train", "validation"]:
        vsmec_file = vsmec_path / f"{split}.csv"
        vigo_file = vigo_path / f"{split}.csv"

        if not vsmec_file.exists():
            print(f"Warning: {vsmec_file} not found, skipping {split}.")
            continue

        if not vigo_file.exists():
            print(f"Warning: {vigo_file} not found. Merging {split} with VSMEC only.")
            include_vigo = False
        else:
            include_vigo = True

        stats = merge_split(
            vsmec_path=vsmec_file,
            vigoemotions_path=vigo_file,
            output_path=out_path / f"{split}.csv",
            include_vigoemotions=include_vigo,
            single_label_only=True,
        )
        all_stats[split] = stats
        print_merge_report(split, stats)

    # test: VSMEC only (copy)
    test_src = vsmec_path / "test.csv"
    test_dst = out_path / "test.csv"
    if test_src.exists():
        shutil.copy2(test_src, test_dst)
        test_df = pd.read_csv(test_dst)
        print(f"\n{'='*60}")
        print(f"  TEST SPLIT: copied VSMEC test as-is ({len(test_df)} samples)")
        print(f"  → {test_dst}")
        dist = Counter(test_df["Emotion"].tolist())
        total = sum(dist.values())
        for cls in VSMEC_CLASSES:
            cnt = dist.get(cls, 0)
            print(f"    {cls:12s}: {cnt:5d} ({cnt/total*100:5.1f}%)")
    else:
        print(f"Warning: {test_src} not found. Test split not created.")

    # Final summary
    print(f"\n{'='*60}")
    print("  MERGE COMPLETE")
    print(f"{'='*60}")
    for split, stats in all_stats.items():
        print(f"  {split:12s}: {stats.get('final_count', 0):6d} samples")

    # Augmentation suggestions
    if "train" in all_stats:
        _suggest_augment_targets(all_stats["train"])

    print(f"\nNext step: python -m src.data.preprocess --input-dir {output_dir}")


if __name__ == "__main__":
    main()
