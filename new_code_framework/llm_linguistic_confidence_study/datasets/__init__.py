from datasets import load_dataset
import os
import pandas as pd

def load_dataset(dataset_name, dataset_url, dataset_dir=None):
    if dataset_name == "simple_qa":
        if os.path.exists(dataset_dir):
            return pd.read_csv(dataset_dir)
        else: # download the dataset
            df = pd.read_csv(dataset_url)
            df.to_csv(dataset_dir, index=False)
            return df
    elif dataset_name == "mmlu_pro":
        pass
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")