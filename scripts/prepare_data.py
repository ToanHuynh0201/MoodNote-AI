"""
Chạy toàn bộ data pipeline: download → merge → preprocess → resplit → augment
Chạy local để chuẩn bị data; Colab chỉ cần load processed data và train.

Cách dùng:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --hf-token hf_xxx
    python scripts/prepare_data.py --skip-download   # nếu đã có data/raw/
    python scripts/prepare_data.py --skip-augment    # bỏ qua bước augmentation
"""
import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Prepare training data (local)")
    parser.add_argument("--hf-token", default=None,
                        help="HuggingFace token cho ViGoEmotions (hoặc set env HF_TOKEN)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Bỏ qua bước download (dùng khi đã có data/raw/)")
    parser.add_argument("--skip-augment", action="store_true",
                        help="Bỏ qua bước augmentation")
    args = parser.parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    raw_dir         = str(REPO_ROOT / "data" / "raw")
    merged_dir      = str(REPO_ROOT / "data" / "merged")
    processed_dir   = str(REPO_ROOT / "data" / "processed")
    config_path     = str(REPO_ROOT / "configs" / "model_config.yaml")
    augmented_train = str(REPO_ROOT / "data" / "processed" / "train_augmented.csv")

    if not args.skip_download:
        # ── Step 1: Download UIT-VSMEC ────────────────────────────────────────
        print("\n" + "=" * 60)
        print("Step 1/5: Downloading UIT-VSMEC...")
        print("=" * 60)
        from src.data.download_dataset import download_uit_vsmec
        download_uit_vsmec(output_dir=raw_dir)

        # ── Step 2: Download ViGoEmotions ─────────────────────────────────────
        print("\n" + "=" * 60)
        print("Step 2/5: Downloading ViGoEmotions...")
        print("=" * 60)
        from src.data.download_vigoemotions import download_vigoemotions
        download_vigoemotions(output_dir=raw_dir, token=hf_token)
    else:
        print("Skipping download (--skip-download)")

    # ── Step 3: Merge ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 3/5: Merging VSMEC + ViGoEmotions...")
    print("=" * 60)
    from src.data.merge_datasets import main as merge_main
    merge_main(
        vsmec_dir=raw_dir,
        vigoemotions_dir=str(REPO_ROOT / "data" / "raw" / "vigoemotions"),
        output_dir=merged_dir,
    )

    # ── Step 4: Preprocess ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 4/5: Preprocessing (word segmentation)...")
    print("=" * 60)
    from src.data.preprocess import preprocess_dataset
    preprocess_dataset(
        input_dir=merged_dir,
        output_dir=processed_dir,
        config_path=config_path,
    )

    # ── Step 5: Resplit stratified ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 5/5: Resplit stratified 80/10/10...")
    print("=" * 60)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "resplit", REPO_ROOT / "scripts" / "resplit_stratified.py"
    )
    resplit = importlib.util.module_from_spec(spec)  # type: ignore
    spec.loader.exec_module(resplit)  # type: ignore

    if args.skip_augment:
        print("\nSkipping augmentation (--skip-augment)")
    else:
        # ── Step 6: Augmentation ──────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("Step 6/5: Augmentation (random deletion/swap/insertion)...")
        print("=" * 60)
        import pandas as pd
        from src.data.augment import augment_dataset

        augment_dataset(
            input_csv=str(REPO_ROOT / "data" / "processed" / "train.csv"),
            output_csv=augmented_train,
            target_counts={0: 2000, 2: 1800, 3: 1500, 4: 1100, 5: 1800},
            techniques=["swap", "insertion"],
            seed=42,
        )

        # Remove any augmented texts that appear in test (leakage prevention)
        test_path = REPO_ROOT / "data" / "processed" / "test.csv"
        if test_path.exists():
            test_texts = set(pd.read_csv(test_path)["text"].str.strip().str.lower())
            aug_df     = pd.read_csv(augmented_train)
            before     = len(aug_df)
            aug_df     = aug_df[~aug_df["text"].str.strip().str.lower().isin(test_texts)]
            n_removed  = before - len(aug_df)
            if n_removed:
                aug_df.to_csv(augmented_train, index=False, encoding="utf-8")
                print(f"Removed {n_removed} augmented samples overlapping with test set.")

    print("\n" + "=" * 60)
    print("Data pipeline hoàn tất!")
    print(f"  Train         : {processed_dir}/train.csv")
    if not args.skip_augment:
        print(f"  Train augmented: {augmented_train}")
    print(f"  Val           : {processed_dir}/validation.csv")
    print(f"  Test          : {processed_dir}/test.csv")
    print("=" * 60)
    print("\nBước tiếp theo:")
    print("  Upload thư mục data/processed/ lên Google Drive tại:")
    print("  MyDrive/MoodNote-AI/processed/")
    print("  Sau đó chạy train_colab.ipynb trên Colab.")


if __name__ == "__main__":
    main()
