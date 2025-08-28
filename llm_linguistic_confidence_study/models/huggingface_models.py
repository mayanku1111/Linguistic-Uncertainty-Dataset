from omegaconf import DictConfig, ListConfig
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

class Huggingface():
    def __init__(self, model_cfg: DictConfig):
        self.model_cfg = model_cfg
        self.model, self.tokenizer = self.load_model(self.model_cfg)
        self.generate_config = self.get_generate_config(self.model_cfg)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __call__(self, prompts: list[str], task_name: str = None, batch_job_id: ListConfig | str = None) -> list[str]:
        responses = []
        batch_size = 4  
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            outputs = self.model.generate(
                **inputs,
                **self.generate_config,
                # max_new_tokens=128
            )
            responses.extend(self.tokenizer.batch_decode(outputs, skip_special_tokens=True))

        return responses
    
    def load_model(self, cfg):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if not os.path.exists(cfg.save_path):
            tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_id)
            model = AutoModelForCausalLM.from_pretrained(
                cfg.base_model_id,
                device_map=device,
            )
            # Optionally apply LoRA if lora_weight_path provided
            if hasattr(cfg, "lora_weight_path") and cfg.lora_weight_path is not None:
                model = PeftModel.from_pretrained(model, cfg.lora_weight_path)
                model = model.merge_and_unload()
            # model.save_pretrained(cfg.save_path)
            # tokenizer.save_pretrained(cfg.save_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(cfg.save_path)
            model = AutoModelForCausalLM.from_pretrained(
                cfg.save_path,
                device_map=device,
            )
        return model, tokenizer

    def get_generate_config(self, cfg):
        return {
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "top_k": cfg.top_k,
            "min_p": cfg.min_p,
            "max_new_tokens": cfg.max_tokens,
        }

if __name__ == "__main__":
    aaa = Huggingface(r"llm_linguistic_confidence_study/configs/qa_model/huggingface.yaml")
    aaa(["What is the capital of Australia?", "Who was awarded the Oceanography Society's Jerlov Award in 2018?"])