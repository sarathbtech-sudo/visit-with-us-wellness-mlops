import os
from pathlib import Path

import pandas as pd
from datasets import Dataset
from huggingface_hub import HfApi, HfFolder


def main():
    """
    Load tourism.csv from tourism_project/data and push it
    as a dataset to Hugging Face Hub using env vars:

    - HF_TOKEN
    - HF_DATASET_ID
    """
    data_path = Path("tourism_project/data/tourism.csv")
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} not found. Make sure tourism.csv is committed in the repo."
        )

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print("Shape:", df.shape)

    ds = Dataset.from_pandas(df)

    hf_token = os.environ["HF_TOKEN"]
    dataset_id = os.environ["HF_DATASET_ID"]

    # Save token locally (required for some HF operations)
    HfFolder.save_token(hf_token)

    api = HfApi(token=hf_token)
    print(f"Pushing dataset to Hugging Face Hub: {dataset_id} ...")
    ds.push_to_hub(dataset_id, token=hf_token)

    print("✅ Dataset registration completed.")


if __name__ == "__main__":
    main()
