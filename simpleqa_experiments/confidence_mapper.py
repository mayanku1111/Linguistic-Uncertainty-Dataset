import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# === model definition ===
class UncertaintyRegressor(nn.Module):
    def __init__(self, model_name='distilroberta-base'):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.reg_head = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 1),
            nn.Sigmoid()  # output range in [0, 1]
        )

    def forward(self, input_ids, attention_mask):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_hidden = output.last_hidden_state[:, 0]
        return self.reg_head(cls_hidden).squeeze(-1)
    
if __name__ == "__main__":
    #########################################################
    # example
    #########################################################

    # === initialize model ===
    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
    model = UncertaintyRegressor()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load("tmp/uncertainty_model_human_anno_valid_count_2and1.pth"))

    # === example data ===
    results = [
        "I don't have that specific winner memorized — would you like me to look it up now?",
        "The women's liberal arts college in Cambridge, Massachusetts, was Radcliffe College."
    ]

    inputs = tokenizer(results, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.eval()
    with torch.no_grad():
        pred_scores = model(inputs["input_ids"], inputs["attention_mask"]).squeeze().cpu().numpy()

    print(pred_scores)