"""Load the Bitext customer support dataset into a pandas DataFrame."""

from pathlib import Path
from typing import Optional

import pandas as pd

HF_CSV_URL = (
    "https://huggingface.co/datasets/bitext/"
    "Bitext-customer-support-llm-chatbot-training-dataset/resolve/main/"
    "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LOCAL_CSV_PATH = DATA_DIR / "bitext_dataset.csv"

_cached_df: Optional[pd.DataFrame] = None


def load_dataset(force_download: bool = False) -> pd.DataFrame:
    """Return the Bitext dataset as a pandas DataFrame.

    On first call, downloads the CSV directly from Hugging Face and caches it
    locally. Subsequent calls read from the local cache. The DataFrame is also
    memoized in-process so repeated calls within a single run don't re-read
    from disk.

    Args:
        force_download: If True, re-download from Hugging Face even if a
            local cache exists.

    Returns:
        A DataFrame with columns: flags, instruction, category, intent, response.
    """
    global _cached_df

    if _cached_df is not None and not force_download:
        return _cached_df

    if force_download or not LOCAL_CSV_PATH.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Downloading dataset from {HF_CSV_URL} ...")
        df = pd.read_csv(HF_CSV_URL)
        df.to_csv(LOCAL_CSV_PATH, index=False)
        print(f"Saved to {LOCAL_CSV_PATH}")
    else:
        df = pd.read_csv(LOCAL_CSV_PATH)

    _cached_df = df
    return df


def dataset_summary(df: pd.DataFrame) -> dict:
    """Return a short summary of the dataset for sanity checks."""
    return {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "columns": list(df.columns),
        "num_categories": df["category"].nunique(),
        "categories": sorted(df["category"].unique().tolist()),
        "num_intents": df["intent"].nunique(),
        "intents": sorted(df["intent"].unique().tolist()),
    }


if __name__ == "__main__":
    df = load_dataset()
    summary = dataset_summary(df)
    print(f"Rows: {summary['num_rows']:,}")
    print(f"Columns ({summary['num_columns']}): {summary['columns']}")
    print(f"\nCategories ({summary['num_categories']}):")
    for cat in summary["categories"]:
        print(f"  - {cat}")
    print(f"\nIntents ({summary['num_intents']}):")
    for intent in summary["intents"]:
        print(f"  - {intent}")
    print("\nSample row:")
    sample = df.iloc[0]
    for col in df.columns:
        value = str(sample[col])
        if len(value) > 100:
            value = value[:100] + "..."
        print(f"  {col}: {value}")
