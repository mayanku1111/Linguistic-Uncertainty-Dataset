import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
import pandas as pd
import numpy as np

# === 示例数据 ===
sample_data = [
    {"text": "I think this might be correct.", "score": 0.7},
    {"text": "It is definitely true.", "score": 0.1},
    {"text": "I'm not sure if this will work.", "score": 0.9},
    {"text": "This is probably accurate.", "score": 0.6},
    {"text": "There is a chance it could fail.", "score": 0.8},
]

import pickle
# with open("tmp/llm_anno_data.pkl", "rb") as f:
#     sample_data = pickle.load(f)  # load llm annotation data

with open("tmp/human_anno_data_valid_count_2and1.pkl", "rb") as f:
    sample_data = pickle.load(f)  # load human annotation data

# === 数据集类 ===
class UncertaintyDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=32):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoded = self.tokenizer(
            item["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "score": torch.tensor(item["score"], dtype=torch.float)
        }

# === 模型定义 ===
class UncertaintyRegressor(nn.Module):
    def __init__(self, model_name='distilroberta-base'):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.reg_head = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 1),
            nn.Sigmoid()  # 输出范围在 [0, 1]
        )

    def forward(self, input_ids, attention_mask):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_hidden = output.last_hidden_state[:, 0]
        return self.reg_head(cls_hidden).squeeze(-1)

# === 初始化 ===
tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
model = UncertaintyRegressor()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.load_state_dict(torch.load("tmp/uncertainty_model_human_anno_valid_count_2and1.pth"))

dataset = UncertaintyDataset(sample_data, tokenizer)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

test_sentences = [
"I don't have that specific winner memorized — would you like me to look it up now?",
"The women's liberal arts college in Cambridge, Massachusetts, was Radcliffe College."
]

from openai import OpenAI
import json

# Initialize OpenAI client with API key from environment variable
client = OpenAI(api_key="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")

# batch_689da7dd35708190a6d0ed4012152e50
# batch_689da7e0394881908ea6ad1d2bb7a249
# batch_689da7e30b008190a9768c88d235d896
batch_job = client.batches.retrieve("batch_689da7dd35708190a6d0ed4012152e50")

# Get the output file content
result_bytes = client.files.content(batch_job.output_file_id).content
result_text  = result_bytes.decode('utf-8')

results = []
for line in result_text.strip().split("\n"):
    entry = json.loads(line)
    idx   = int(entry["custom_id"])
    answer = entry["response"]["body"]["choices"][0]["message"]["content"].strip()
    results.append(answer)

inputs = tokenizer(results, return_tensors="pt", truncation=True, padding=True, max_length=128)
inputs = {k: v.to(device) for k, v in inputs.items()}

model.eval()

total_length = 0
with torch.no_grad():
    pred_scores = model(inputs["input_ids"], inputs["attention_mask"]).squeeze().cpu().numpy()
    for i in range(len(results)):
        print(results[i], pred_scores[i])
        total_length += len(results[i])

print(total_length)