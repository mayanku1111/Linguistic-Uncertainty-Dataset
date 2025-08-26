from .LoRA import LoRA

from omegaconf import DictConfig


class Finetune:
    def __init__(self, fintune_cfg: DictConfig, base_qa_model_cfg: DictConfig):
        self.fintune_cfg = fintune_cfg
        self.base_qa_model_cfg = base_qa_model_cfg
        self.finetune = self.get_finetune_method(fintune_cfg.name)

    def __call__(self, dataset, qa_batch_job_id: str = None, grader_batch_job_id: str = None):
        return self.get_finetune_method(dataset, qa_batch_job_id, grader_batch_job_id)

    def get_finetune_method(self, finetune_method_name):
        if finetune_method_name == "lra":
            return LoRA(self.fintune_cfg, self.base_qa_model_cfg)
        elif finetune_method_name == "sft":
            pass
            # return PTrueConfidenceExtractor(self.confidence_extraction_method_cfg, self.qa_model_cfg)
        elif finetune_method_name == "all":
            pass
            # return SemanticUncertaintyExtractor(self.confidence_extraction_method_cfg, self.qa_model_cfg)
        else:
            raise ValueError(
                f"Invalid finetune method: {finetune_method_name}")
