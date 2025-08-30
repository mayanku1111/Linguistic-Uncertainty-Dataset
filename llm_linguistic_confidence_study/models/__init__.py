from omegaconf import DictConfig

# Import optional model backends. If a dependency is missing, keep the symbol as None
# so importing this package doesn't fail for users who only need a subset.
try:
    from .openai_models import GPT
except Exception:
    GPT = None

try:
    from .togetherai_models import TogetherAI
except Exception:
    TogetherAI = None

try:
    from .xai_models import Grok
except Exception:
    Grok = None

try:
    from .anthropic_models import Claude
except Exception:
    Claude = None

try:
    from .huggingface_models import Huggingface
except Exception:
    Huggingface = None

try:
    from .openrouter_llama import OpenRouterLlama
except Exception:
    # If the full class couldn't be imported (maybe due to typing or other runtime issues),
    # try to import the lightweight helper and build a fallback wrapper so the package
    # can still be used.
    try:
        from .openrouter_llama import generate_openrouter_llama

        class OpenRouterLlama:
            def __init__(self, model_cfg: DictConfig):
                # minimal mapping from cfg to expected attributes
                try:
                    self.model_name = model_cfg.name
                except Exception:
                    self.model_name = model_cfg.get("name", model_cfg.get("base_model_id", "meta-llama/llama-3.1-8b-instruct"))
                self.api_key = getattr(model_cfg, "api_key", None) or None
                self.temperature = getattr(model_cfg, "temperature", None)
                self.top_p = getattr(model_cfg, "top_p", None)
                self.top_k = getattr(model_cfg, "top_k", None)
                self.max_tokens = getattr(model_cfg, "max_tokens", None)

            def __call__(self, prompts, task_name=None, batch_job_id=None):
                if isinstance(prompts, str):
                    prompts = [prompts]
                results = []
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
                    except Exception:
                        text = None
                    results.append(text)
                return results
    except Exception:
        OpenRouterLlama = None
        OPENROUTER_IMPORT_ERROR = True
    else:
        OPENROUTER_IMPORT_ERROR = False

OPEN_AI_MODEL_LIST = [
    "gpt-5", 
    "gpt-5-mini",
    "gpt-5-nano",
]

# this is a list of models that support batch inference
TOGETHER_AI_MODEL_LIST = [
    "deepseek-ai/DeepSeek-R1-0528-tput",
    "deepseek-ai/DeepSeek-V3",
    "meta-llama/Llama-3-70b-chat-hf",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "moonshotai/Kimi-K2-Instruct",
    "Qwen/Qwen2.5-72B-Instruct-Turbo",
    "Qwen/Qwen2.5-7B-Instruct-Turbo",
    "Qwen/Qwen3-235B-A22B-fp8-tput",
    "Qwen/QwQ-32B",
    "openai/whisper-large-v3",
    "deepseek-ai/DeepSeek-R1",
    "google/gemma-3n-E4B-it",
    "marin-community/marin-8b-instruct",
    "meta-llama/Meta-Llama-3-70B-Instruct-Turbo",
    "Qwen/Qwen2.5-VL-72B-Instruct",
    "Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
    "togethercomputer/Refuel-Llm-V2",
    "togethercomputer/Refuel-Llm-V2-Small",
    "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
    "openai/gpt-oss-120b",
    "zai-org/GLM-4.5-Air-FP8",
    "Qwen/Qwen3-235B-A22B-Thinking-2507",
    "openai/gpt-oss-20b",
]

ANTHROPIC_AI_MODEL_LIST = [
    "claude-opus-4-1-20250805",
    "claude-opus-4-20250514",
    "claude-sonnet-4-20250514",
    "claude-3-7-sonnet-20250219",
    "claude-3-5-sonnet-20240620",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
    "claude-3-haiku-20240307",
]

# Grok does not support batch inference; real-time inference is used
X_AI_MODEL_LIST = [
    "grok-4-0709",
    "grok-3",
    "grok-3-mini",
]


HUGGING_FACE_LIST = [
    "Qwen/Qwen3-8B-uncertainty",
    "meta-llama/Meta-Llama-3.1-8B-Instruct"
]

OPENROUTER_LLAMA_LIST = [
    "meta-llama/llama-3.1-8b-instruct"
]

class LLM:
    def __init__(self, model_cfg: DictConfig):
        self.model_cfg = model_cfg
        self.model: GPT | TogetherAI | Grok | Claude | Huggingface = self.prepare_model(model_cfg)


    def __call__(self, prompts: list[str], task_name: str, batch_job_id: list[str] | str = None) -> list[str]:
        responses = self.model(prompts, task_name, batch_job_id)
        return responses


    def prepare_model(self, model_cfg: DictConfig):
        if model_cfg.name in OPEN_AI_MODEL_LIST:
            if GPT is None:
                raise RuntimeError("Requested OpenAI model but `openai` backend is not available (missing dependency).")
            return GPT(model_cfg)
        elif model_cfg.name in TOGETHER_AI_MODEL_LIST:
            if TogetherAI is None:
                raise RuntimeError("Requested TogetherAI model but `togetherai` backend is not available (missing dependency).")
            return TogetherAI(model_cfg)
        elif model_cfg.name in ANTHROPIC_AI_MODEL_LIST:
            if Claude is None:
                raise RuntimeError("Requested Anthropic model but `anthropic` backend is not available (missing dependency).")
            return Claude(model_cfg)
        elif model_cfg.name in X_AI_MODEL_LIST:
            if Grok is None:
                raise RuntimeError("Requested Grok model but `xai` backend is not available (missing dependency).")
            return Grok(model_cfg)
        elif model_cfg.name in HUGGING_FACE_LIST:
            if Huggingface is None:
                raise RuntimeError("Requested HuggingFace model but `huggingface` backend is not available (missing dependency).")
            return Huggingface(model_cfg)
        elif model_cfg.name in OPENROUTER_LLAMA_LIST:
            if OpenRouterLlama is None:
                raise RuntimeError("Requested OpenRouter Llama model but the OpenRouter wrapper failed to import.")
            return OpenRouterLlama(model_cfg)
        else:
            raise ValueError(f"Invalid model name: {model_cfg.name}")
        