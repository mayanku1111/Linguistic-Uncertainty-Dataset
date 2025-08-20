from datasets import load_dataset
import os
import pandas as pd

def load_dataset(dataset_cfg):
    if dataset_cfg.name == "simple_qa":
        if os.path.exists(dataset_cfg.dir):
            return pd.read_csv(dataset_cfg.dir)
        else: # download the dataset
            df = pd.read_csv(dataset_cfg.url)
            df.to_csv(dataset_cfg.dir, index=False)
            return df
    elif dataset_cfg.name == "mmlu_pro":
        pass
    else:
        raise ValueError(f"Invalid dataset name: {dataset_cfg.name}")