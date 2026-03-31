import pandas as pd
import numpy as np

# Load merged data
merged_train = pd.read_csv("d:/Project/MoodNote/MoodNote-AI/data/processed/train.csv", encoding='utf-8')
merged_test = pd.read_csv("d:/Project/MoodNote/MoodNote-AI/data/processed/test.csv", encoding='utf-8')
merged_val = pd.read_csv("d:/Project/MoodNote/MoodNote-AI/data/processed/validation.csv", encoding='utf-8')

# Combine all
all_data = pd.concat([merged_train, merged_val, merged_test], ignore_index=True).reset_index(drop=True)
print(f"Total samples: {len(all_data)}")

# Stratified split manually
np.random.seed(42)
all_data['_split'] = np.nan
all_data['_rand'] = np.random.random(len(all_data))

for label in range(7):
    mask = all_data['label'] == label
    indices = np.where(mask)[0]
    n = len(indices)
    
    # Sort by random and assign: 80% train, 10% val, 10% test
    rand_sorted = np.argsort(all_data.loc[indices, '_rand'].to_numpy())
    train_n = int(n * 0.8)
    val_n = int(n * 0.1)
    
    all_data.loc[indices[rand_sorted[:train_n]], '_split'] = 'train'
    all_data.loc[indices[rand_sorted[train_n:train_n+val_n]], '_split'] = 'val'
    all_data.loc[indices[rand_sorted[train_n+val_n:]], '_split'] = 'test'

train = all_data[all_data['_split'] == 'train'].drop(['_split', '_rand'], axis=1).reset_index(drop=True)
val = all_data[all_data['_split'] == 'val'].drop(['_split', '_rand'], axis=1).reset_index(drop=True)
test = all_data[all_data['_split'] == 'test'].drop(['_split', '_rand'], axis=1).reset_index(drop=True)

print(f"\nNew splits:")
print(f"Train: {len(train)} samples")
print(f"Val:   {len(val)} samples")
print(f"Test:  {len(test)} samples")

print("\n=== NEW DISTRIBUTION (Stratified) ===")
for label in range(7):
    train_pct = (train['label'] == label).sum() / len(train) * 100
    val_pct = (val['label'] == label).sum() / len(val) * 100
    test_pct = (test['label'] == label).sum() / len(test) * 100
    print(f"Label {label}: Train {train_pct:5.1f}% | Val {val_pct:5.1f}% | Test {test_pct:5.1f}%")

# Save
train.to_csv("d:/Project/MoodNote/MoodNote-AI/data/processed/train.csv", index=False, encoding='utf-8')
val.to_csv("d:/Project/MoodNote/MoodNote-AI/data/processed/validation.csv", index=False, encoding='utf-8')
test.to_csv("d:/Project/MoodNote/MoodNote-AI/data/processed/test.csv", index=False, encoding='utf-8')

print("\nFiles saved!")
