import torch
import torch.nn as nn
import pandas as pd
from omegaconf import DictConfig
from llm_linguistic_confidence_study.models import LLM
from llm_linguistic_confidence_study.confidence_extraction_methods.linguistic_confidence import LinguisticConfidenceEstimator

LORA_TEST = """
{question}
""".strip()


class test_LoRA():
    def __init__(self, cfg, qa_model_cfg):
        self.confidence_mapper = self.get_confidence_mapper(self.cfg)

    def __call__(self, dataset, qa_batch_job_id: str = None, grader_batch_job_id: str = None):
        if dataset.name == "simple_qa" or dataset.name == "mini_simple_qa":
            qa_responses = self.generate_qa_responses(
                dataset.df, self.cfg, task_name=f"simple_qa_su_qa", qa_batch_job_id=qa_batch_job_id)
            # combine qa_responses and dataset_df
            response_df = dataset.df.copy()
            response_df["responses"] = qa_responses
            # confidence estimation
            confidences = self.confidence_mapper(response_df)
            response_df["confidences"] = confidences
            # grade the accuracy of the confidence scores
            accuracies = dataset.grade_responses(
                response_df["responses"], grader_batch_job_id=grader_batch_job_id, task_name=f"simple_qa_lc_grader")
            response_df["accuracies"] = accuracies
        elif dataset.name == "mmlu_pro":
            pass
        else:
            raise ValueError(f"Invalid dataset name: {dataset.name}")
        # return the response_df
        return response_df

    def get_qa_model(self, qa_model_cfg):
        return LLM(qa_model_cfg)

    def get_confidence_mapper(self, cfg):
        if cfg.mapper_name == "self-trained":
            confidence_estimator = LinguisticConfidenceEstimator(cfg)
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
            confidence_estimator.reg_head.load_state_dict(
                torch.load(cfg.state_dict_path))
            return confidence_estimator.to(device)
        else:
            raise ValueError(
                f"Invalid confidence extraction method: {cfg}")

    def generate_qa_responses(self, dataset_df: pd.DataFrame, cfg: DictConfig, task_name: str, qa_batch_job_id: str = None) -> list[str]:
        # prepare prompts
        if cfg.qa_template == "test":
            prompt_template = LORA_TEST
        else:
            raise ValueError(f"Invalid qa template: {cfg.qa_template}")
        qa_prompts = [prompt_template.format(
            question=row["problem"]) for _, row in dataset_df.iterrows()]
        # generate responses
        responses = self.qa_model(
            qa_prompts, task_name=task_name, batch_job_id=qa_batch_job_id)
        # post-process the responses if needed
        return responses
