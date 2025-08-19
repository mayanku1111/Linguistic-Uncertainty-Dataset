from openai import OpenAI

from .custom_types import *
from .framework_config import *

class OpenAIModel(ModelBase):
    def __init__(self, model_id: str):
        self.model_id = model_id                               # used for api calling
        self.model_name = MODEL_NAME_LIST[model_id]            # used for file name/display purposes
        self.api_key = OPENAI_API_KEY
        

    def batch_inference(self, prompt_list: list[str]) -> list[str]:
        pass


    def real_time_inference(self, prompt_list: list[str]) -> list[str]:
        pass