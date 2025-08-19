import pandas as pd

class ModelBase():
    model_id: str
    model_name: str

class EvalBase():
    target_model_id: ModelBase 
    grader_model_id: ModelBase 
    qa_data: pd.DataFrame 
    data_set: str
    results: pd.DataFrame 