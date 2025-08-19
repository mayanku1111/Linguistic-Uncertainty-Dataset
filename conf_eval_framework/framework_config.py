import os

DEFAULT_TARGET_MODEL = "gpt-5-mini"
DEFAULT_GRADER_MODEL = "gpt-5-mini"

CONFIDENCE_METHODS = ["verbal", "linguistic", "p_true", "semantic_entropy", "se", "pt", "ling", "vb"]

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# model_id : model_name
MODEL_NAME_LIST = {
    "gpt-5-mini": "gpt-5-mini",
    "gpt-4": "gpt-4",
}