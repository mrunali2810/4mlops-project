"""Clean raw data and create train/test splits."""
import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess(input_path="data/data.csv", out_dir="data/processed",
               test_size=0.2, seed=42):
    df = pd.read_csv(input_path).dropna().drop_duplicates()
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["target"]
    )
    os.makedirs(out_dir, exist_ok=True)
    train_df.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, "test.csv"), index=False)
    print(f"Preprocessed: train={len(train_df)} test={len(test_df)}")
    return train_df, test_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/data.csv")
    parser.add_argument("--out", default="data/processed")
    args = parser.parse_args()
    preprocess(args.input, args.out)
