from omegaconf import DictConfig, ListConfig
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

from typing import Tuple

def _load_tokenizer_and_model(model_id: str, use_save_path: str | None, device: str) -> Tuple:
    """Helper to load tokenizer and model with clearer errors for gated/private repos.

    Raises RuntimeError with actionable instructions if download fails (e.g. 401 or private repo).
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device,
        )
        return model, tokenizer
    except Exception as e:
        # Provide an actionable error for common HF gated/private repo errors.
        msg = (
            f"Failed to load Hugging Face model/tokenizer for '{model_id}': {e}\n"
            "This commonly happens when the model repository is gated or private.\n"
            "Fixes:\n"
            "  1) Set the environment variable HUGGINGFACE_HUB_TOKEN to a token with access:\n"
            "       export HUGGINGFACE_HUB_TOKEN=\"hf_xxx...\"\n"
            "     or run: huggingface-cli login\n"
            "  2) If you don't have access, use an alternative runtime (OpenRouter) and an OpenRouter model id in your config.\n"
            "  3) Or place a local copy of the model/tokenizer and set `cfg.save_path` to that folder.\n"
            "Original exception:\n" + repr(e)
        )
        raise RuntimeError(msg)

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
        if cfg.save_path is not None and not os.path.exists(cfg.save_path):
            model, tokenizer = _load_tokenizer_and_model(cfg.base_model_id, cfg.save_path, device)
            # Optionally apply LoRA if lora_weight_path provided
            if hasattr(cfg, "lora_weight_path") and cfg.lora_weight_path is not None:
                model = PeftModel.from_pretrained(model, cfg.lora_weight_path)
                model = model.merge_and_unload()
            # model.save_pretrained(cfg.save_path)
            # tokenizer.save_pretrained(cfg.save_path)
        elif cfg.save_path is not None:
            # load from a local save path
            try:
                tokenizer = AutoTokenizer.from_pretrained(cfg.save_path)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                model = AutoModelForCausalLM.from_pretrained(
                    cfg.save_path,
                    device_map=device,
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load model from local save_path '{cfg.save_path}': {e}")
        else:
            model, tokenizer = _load_tokenizer_and_model(cfg.base_model_id, None, device)
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