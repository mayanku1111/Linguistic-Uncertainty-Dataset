import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


SIMPLE_QA_EVAL_VANILLA_TEMPLATE = """
Answer the following question using a succinct (at most one sentence) and full answer.

Question: {question}
Answer:
""".strip()

SIMPLE_QA_EVAL_VANILLA_UNCERTAINTY_TEMPLATE = """
Answer the following question using a succinct (at most one sentence) and full answer. If you are uncertain about your answer to the question, convey this uncertainty linguistically by precisely hedging this answer.

Question: {question}
Answer:
""".strip()


class LinguisticConfidenceExtractor():
    def __init__(self, confidence_extraction_method_cfg, dataset_cfg, model_cfg):
        self.confidence_extraction_method_cfg = confidence_extraction_method_cfg
        self.dataset_cfg = dataset_cfg
        self.model_cfg = model_cfg
        self.confidence_estimator = self.get_confidence_estimator(confidence_extraction_method_cfg.name)
        
    def __call__(self, dataset_df, model_cfg):
        prompt = self.prepare_prompt()
        responses = self.generate_responses(prompt, dataset_df, model_cfg)
        df = self.confidence_estimate(responses)
        return df
    
    def prepare_prompt(self):
        pass
    
    def generate_responses(self, prompt, dataset_df, model_cfg):
        pass
    
    def confidence_estimate(self, responses):
        # responses: list of strings
        # return a list of confidence scores in [0, 1]
        inputs = self.tokenizer(responses, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        self.confidence_estimator.eval()
        with torch.no_grad():
            pred_scores = self.confidence_estimator(inputs["input_ids"], inputs["attention_mask"]).squeeze().cpu().numpy()
        return pred_scores
    
    def get_confidence_estimator(self, confidence_extraction_method_name):
        if confidence_extraction_method_name == "linguistic_confidence":
            return LinguisticConfidenceEstimator()
        else:
            raise ValueError(f"Invalid confidence extraction method: {confidence_extraction_method_name}")
        
class LinguisticConfidenceEstimator():
    def __init__(self, model_name='distilroberta-base'):
        self.encoder = AutoModel.from_pretrained(model_name)
        self.reg_head = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 1),
            nn.Sigmoid()  # output range in [0, 1]
        )
        self.load_state_dict(torch.load("tmp/uncertainty_model_human_anno_valid_count_2and1.pth"))

    def __call__(self, input_ids, attention_mask):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_hidden = output.last_hidden_state[:, 0]
        return self.reg_head(cls_hidden).squeeze(-1)
    
    
if __name__ == "__main__":
    #########################################################
    # example
    #########################################################

    # === initialize model ===
    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
    model = LinguisticConfidenceEstimator()
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
    
