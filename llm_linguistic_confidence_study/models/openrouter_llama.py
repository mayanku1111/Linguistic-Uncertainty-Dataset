import os
import requests
from omegaconf import DictConfig


import requests
import os


def generate_openrouter_llama(prompt, api_key=None, model_name="meta-llama/llama-3.1-8b-instruct", **gen_kwargs):
    """Simple helper that calls OpenRouter chat completions and returns the assistant text.

    gen_kwargs may contain temperature, top_p, top_k, max_tokens, etc. Those will be forwarded
    into the request body when present.
    """
    api_key = api_key or os.getenv("OPENROUTER_API_KEY")
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    # forward generation kwargs if provided
    for k, v in gen_kwargs.items():
        if v is not None:
            data[k] = v

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise RuntimeError(f"OpenRouter API error: {response.status_code} {response.text}")


class OpenRouterLlama:
    """Lightweight model wrapper for OpenRouter-hosted Llama models.

    Implements the same call signature expected by the repo's `LLM` wrapper: it accepts a
    DictConfig in the constructor and is callable with a list of prompts.
    """
    def __init__(self, model_cfg: DictConfig):
        # model_cfg is expected to have fields like `name` (model id) and optional generation params
        try:
            self.model_name = model_cfg.name
        except Exception:
            # DictConfig behaves like an object; fall back to dict-get
            self.model_name = model_cfg.get("name", model_cfg.get("base_model_id", "meta-llama/llama-3.1-8b-instruct"))

        # API key can come from config or environment
        self.api_key = getattr(model_cfg, "api_key", None) or os.getenv("OPENROUTER_API_KEY")

        # generation params
        self.temperature = getattr(model_cfg, "temperature", None)
        self.top_p = getattr(model_cfg, "top_p", None)
        self.top_k = getattr(model_cfg, "top_k", None)
        self.max_tokens = getattr(model_cfg, "max_tokens", None)

    def __call__(self, prompts: list[str], task_name: str = None, batch_job_id: list[str] | str = None) -> list[str]:
        """Call the OpenRouter chat completions endpoint for each prompt and return assistant texts.

        This is intentionally simple (synchronous, one request per prompt). It returns a list of
        strings aligned with `prompts`. On API error, the corresponding entry will be None.
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        responses = []
        for p in prompts:
            try:
                text = generate_openrouter_llama(
                    p,
                    api_key=self.api_key,
                    model_name=self.model_name,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    max_tokens=self.max_tokens,
                )
            except Exception as e:
                # keep behaviour non-fatal: append None so callers can handle missing outputs
                text = None
            responses.append(text)

        return responses
