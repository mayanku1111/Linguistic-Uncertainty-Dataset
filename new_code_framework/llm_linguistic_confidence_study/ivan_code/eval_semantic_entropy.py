
import pandas as pd
import os

from .custom_types import *

class SemanticEntropyConfidence(EvalBase):
    def __init__(self, qa_data, target_model_id, grader_model_id):
        self.target_model_id = target_model_id
        self.grader_model_id = grader_model_id
        self.qa_data: pd.DataFrame = qa_data
        self.results = pd.DataFrame()