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

with open("train_mapper/raw_dataset/llm_anno_data.pkl", "rb") as f:
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

dataset = UncertaintyDataset(sample_data, tokenizer)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = nn.MSELoss()


test_set_df = pd.read_csv('train_mapper/raw_dataset/dataset_valid_confidence_score_count_3_eval.csv')
test_sentences = test_set_df['uncertainty_expression'].tolist()

test_inputs = tokenizer(test_sentences, return_tensors="pt", truncation=True, padding=True, max_length=128)
test_inputs = {k: v.to(device) for k, v in test_inputs.items()}

best_test_loss = float('inf')

# === 训练 ===
model.train()
model.reg_head.requires_grad = True
model.encoder.requires_grad = False
for epoch in range(100):
    total_loss = 0
    for index, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        scores = batch["score"].to(device)

        optimizer.zero_grad()
        preds = model(input_ids, attention_mask)
        
        loss = criterion(preds, scores)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if index % 100 == 0:
            print(f"Batch {index} - Loss: {total_loss/(index+1):.4f}")


    model.eval()

    with torch.no_grad():
        pred_scores = model(test_inputs["input_ids"], test_inputs["attention_mask"]).squeeze().cpu().numpy()
    test_loss = criterion(torch.tensor(pred_scores, dtype=torch.float32), torch.tensor((test_set_df['annotation_mean']/100).to_numpy(), dtype=torch.float32))
    print(f"Epoch {epoch+1} - Train Loss: {total_loss/len(dataloader):.4f} - Test Loss: {test_loss:.4f} {'(new best loss)' if test_loss < best_test_loss else ''}")
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        torch.save(model.reg_head.state_dict(), "train_mapper/raw_dataset/llm_anno_trained_reg_head.pth")
        test_set_df['confidence_score_probe_llm_anno_trained'] = pred_scores
        test_set_df.to_csv('train_mapper/raw_dataset/llm_anno_trained.csv', index=False)