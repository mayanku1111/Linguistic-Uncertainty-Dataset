from omegaconf import DictConfig
from .openai_models import GPT

class LLM:
    def __init__(self, model_cfg: DictConfig):
        self.model_cfg = model_cfg
        self.model = self.prepare_model(model_cfg)

    def __call__(self, prompts: list[str], **kwargs) -> list[str]:
        responses = self.model(prompts, **kwargs)
        return responses

    def prepare_model(self, model_cfg: DictConfig):
        if model_cfg.name == "gpt-5-mini":
            return GPT(model_cfg)
        else:
            raise ValueError(f"Invalid model name: {model_cfg.name}")
        
