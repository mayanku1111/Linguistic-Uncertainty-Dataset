from datasets import load_dataset

def load_dataset(dataset_name):
    if dataset_name == "simpleqa":
        return load_dataset("simpleqa")
    elif dataset_name == "mmlu_pro":
        return load_dataset("mmlu_pro")
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")