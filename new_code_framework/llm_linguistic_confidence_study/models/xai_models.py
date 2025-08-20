from openai import OpenAI
from omegaconf import DictConfig
import os

OPENAI_SYSTEM_PROMPT = "You are a helpful assistant."

class Grok():
    def __init__(self, model_cfg: DictConfig):
        self.model_cfg = model_cfg

    def __call__(self, prompts: list[str]) -> list[str]:
        responses = self.real_time_inference(prompts)
        return responses
    
    def real_time_inference(self, prompts: list[str]) -> list[str]:
        client = OpenAI(api_key=os.getenv("XAI_API_KEY"), base_url="https://api.x.ai/v1")
        responses = []
        for prompt in prompts:
            response = client.chat.completions.create(
                model = self.model_cfg.name,
                messages=[
                    {"role": "system", "content": OPENAI_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            responses.append(response.choices[0].message.content.strip())
        return responses
        

        