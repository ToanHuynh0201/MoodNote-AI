import pandas as pd
import numpy as np
from pathlib import Path

# Repo root = two levels up from this script (scripts/ → repo root)
REPO_ROOT    = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

EMOTION_NAMES = {
    0: "Enjoyment", 1: "Sadness", 2: "Anger",
    3: "Fear", 4: "Disgust", 5: "Surprise", 6: "Other"
}

# ── Load train + val only (test giữ nguyên làm held-out benchmark) ────────────
merged_train = pd.read_csv(PROCESSED_DIR / "train.csv",      encoding='utf-8')
merged_val   = pd.read_csv(PROCESSED_DIR / "validation.csv", encoding='utf-8')

all_data = pd.concat([merged_train, merged_val], ignore_index=True)
print(f"Total samples (train+val) before deduplication: {len(all_data)}")

# ── Deduplication ─────────────────────────────────────────────────────────────
# For each unique text:
#   - same text, same label everywhere   → keep 1 copy
#   - same text, different labels, clear majority → keep majority label, 1 copy
#   - same text, different labels, tie           → drop (too ambiguous)

text_groups = all_data.groupby("text")["label"]

resolved_texts = {}   # text → resolved label
n_pure_dups    = 0    # same text, same label (just repeated)
n_majority     = 0    # resolved via majority vote
n_tie_dropped  = 0    # dropped due to tie

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

print(f"\n=== DEDUPLICATION REPORT ===")
print(f"Pure duplicates removed (same label):       {n_pure_dups}")
print(f"Conflicts resolved via majority vote:       {n_majority}")
print(f"Conflicts dropped (tie, ambiguous):         {n_tie_dropped}")
print(f"Total samples after deduplication:          {len(all_data)}")

print(f"\nClass distribution after dedup:")
for label in range(7):
    cnt = (all_data["label"] == label).sum()
    print(f"  {EMOTION_NAMES[label]:12s} ({label}): {cnt:5d}")

# ── Stratified split 89 / 11 (≈ 80/10/10 của tổng ban đầu khi test chiếm ~10%) ──
# Vì test set đã được giữ nguyên từ VSMEC gốc (~10%), phần còn lại split 89/11
# để tỉ lệ train/val/test xấp xỉ 80/10/10 trên toàn bộ data.
np.random.seed(42)
all_data["_rand"]  = np.random.random(len(all_data))
all_data["_split"] = pd.array([None] * len(all_data), dtype=object)

for label in range(7):
    mask    = all_data["label"] == label
    indices = np.where(mask)[0]
    n       = len(indices)

    rand_sorted = np.argsort(all_data.loc[indices, "_rand"].to_numpy())
    train_n = int(n * 0.89)

    all_data.loc[indices[rand_sorted[:train_n]],  "_split"] = "train"
    all_data.loc[indices[rand_sorted[train_n:]], "_split"] = "val"

train = all_data[all_data["_split"] == "train"].drop(columns=["_split", "_rand"]).reset_index(drop=True)
val   = all_data[all_data["_split"] == "val"]  .drop(columns=["_split", "_rand"]).reset_index(drop=True)

# Test set: giữ nguyên file test.csv hiện tại (VSMEC gốc, không bị contaminate)
test = pd.read_csv(PROCESSED_DIR / "test.csv", encoding='utf-8')

print(f"\n=== NEW SPLITS ===")
print(f"Train: {len(train):5d} samples")
print(f"Val:   {len(val):5d} samples")
print(f"Test:  {len(test):5d} samples  (held-out, không thay đổi)")

print(f"\n=== DISTRIBUTION PER SPLIT ===")
header = f"{'Label':<14} {'Train':>6} {'Val':>6} {'Test':>6}"
print(header)
print("-" * len(header))
for label in range(7):
    t = (train["label"] == label).sum()
    v = (val["label"]   == label).sum()
    s = (test["label"]  == label).sum()
    print(f"{EMOTION_NAMES[label]:<14} {t:>6} {v:>6} {s:>6}")

# ── Save (chỉ ghi train + val; test không đổi) ────────────────────────────────
train.to_csv(PROCESSED_DIR / "train.csv",      index=False, encoding="utf-8")
val.to_csv(  PROCESSED_DIR / "validation.csv", index=False, encoding="utf-8")

print("\nFiles saved! (test.csv giữ nguyên)")
