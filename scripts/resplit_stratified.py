"""
Re-split the processed train/validation sets into a stratified 89/11 split
after deduplication, keeping the held-out test set untouched.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Repo root = two levels up from this script (scripts/ → repo root)
REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

sys.path.insert(0, str(REPO_ROOT))
from src.utils.emotion_constants import DEFAULT_EMOTION_LABELS  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("resplit_stratified")
EMOTION_NAMES = DEFAULT_EMOTION_LABELS

# ── Load train + val only (test giữ nguyên làm held-out benchmark) ────────────
merged_train = pd.read_csv(PROCESSED_DIR / "train.csv", encoding="utf-8")
merged_val = pd.read_csv(PROCESSED_DIR / "validation.csv", encoding="utf-8")
test_df = pd.read_csv(PROCESSED_DIR / "test.csv", encoding="utf-8")

# Remove any train/val texts that appear in test (prevents leakage)
test_texts = set(test_df["text"].str.strip().str.lower())
before_leak = len(merged_train) + len(merged_val)
merged_train = merged_train[~merged_train["text"].str.strip().str.lower().isin(test_texts)]
merged_val = merged_val[~merged_val["text"].str.strip().str.lower().isin(test_texts)]
n_leaked = before_leak - len(merged_train) - len(merged_val)
if n_leaked:
    logger.info(
        f"Removed {n_leaked} train/val samples that appeared in test set (leakage prevention)."
    )

all_data = pd.concat([merged_train, merged_val], ignore_index=True)
logger.info(f"Total samples (train+val) before deduplication: {len(all_data)}")

# ── Deduplication ─────────────────────────────────────────────────────────────
# For each unique text:
#   - same text, same label everywhere   → keep 1 copy
#   - same text, different labels, clear majority → keep majority label, 1 copy
#   - same text, different labels, tie           → drop (too ambiguous)

text_groups = all_data.groupby("text")["label"]

resolved_texts = {}  # text → resolved label
n_pure_dups = 0  # same text, same label (just repeated)
n_majority = 0  # resolved via majority vote
n_tie_dropped = 0  # dropped due to tie

for text, labels in text_groups:
    counts = labels.value_counts()
    if len(counts) == 1:
        if labels.count() > 1:
            n_pure_dups += 1
        resolved_texts[text] = int(counts.index[0])
    else:
        top_two = counts.iloc[:2]
        if top_two.iloc[0] > top_two.iloc[1]:
            resolved_texts[text] = int(counts.index[0])
            n_majority += 1
        else:
            n_tie_dropped += 1  # leave out of resolved_texts → will be dropped

# Apply resolved labels and deduplicate
all_data = all_data.drop_duplicates(subset=["text"]).reset_index(drop=True)
all_data = all_data[all_data["text"].isin(resolved_texts)].copy()
all_data["label"] = all_data["text"].map(resolved_texts).astype(int)
all_data = all_data.reset_index(drop=True)

logger.info("=== DEDUPLICATION REPORT ===")
logger.info(f"Pure duplicates removed (same label):       {n_pure_dups}")
logger.info(f"Conflicts resolved via majority vote:       {n_majority}")
logger.info(f"Conflicts dropped (tie, ambiguous):         {n_tie_dropped}")
logger.info(f"Total samples after deduplication:          {len(all_data)}")

logger.info("Class distribution after dedup:")
for label in range(7):
    cnt = (all_data["label"] == label).sum()
    logger.info(f"  {EMOTION_NAMES[label]:12s} ({label}): {cnt:5d}")

# ── Stratified split 89 / 11 (≈ 80/10/10 của tổng ban đầu khi test chiếm ~10%) ──
# Vì test set đã được giữ nguyên từ VSMEC gốc (~10%), phần còn lại split 89/11
# để tỉ lệ train/val/test xấp xỉ 80/10/10 trên toàn bộ data.
np.random.seed(42)
all_data["_rand"] = np.random.random(len(all_data))
all_data["_split"] = pd.array([None] * len(all_data), dtype=object)

for label in range(7):
    mask = all_data["label"] == label
    indices = np.where(mask)[0]
    n = len(indices)

    rand_sorted = np.argsort(all_data.loc[indices, "_rand"].to_numpy())
    train_n = int(n * 0.89)

    all_data.loc[indices[rand_sorted[:train_n]], "_split"] = "train"
    all_data.loc[indices[rand_sorted[train_n:]], "_split"] = "val"

train = (
    all_data[all_data["_split"] == "train"].drop(columns=["_split", "_rand"]).reset_index(drop=True)
)
val = all_data[all_data["_split"] == "val"].drop(columns=["_split", "_rand"]).reset_index(drop=True)

# Test set: giữ nguyên file test.csv hiện tại (VSMEC gốc, không bị contaminate)
test = pd.read_csv(PROCESSED_DIR / "test.csv", encoding="utf-8")

logger.info("=== NEW SPLITS ===")
logger.info(f"Train: {len(train):5d} samples")
logger.info(f"Val:   {len(val):5d} samples")
logger.info(f"Test:  {len(test):5d} samples  (held-out, không thay đổi)")

logger.info("=== DISTRIBUTION PER SPLIT ===")
logger.info(f"{'Label':<14} {'Train':>6} {'Val':>6} {'Test':>6}")
for label in range(7):
    t = (train["label"] == label).sum()
    v = (val["label"] == label).sum()
    s = (test["label"] == label).sum()
    logger.info(f"{EMOTION_NAMES[label]:<14} {t:>6} {v:>6} {s:>6}")

# ── Save (chỉ ghi train + val; test không đổi) ────────────────────────────────
train.to_csv(PROCESSED_DIR / "train.csv", index=False, encoding="utf-8")
val.to_csv(PROCESSED_DIR / "validation.csv", index=False, encoding="utf-8")

logger.info("Files saved! (test.csv giữ nguyên)")
