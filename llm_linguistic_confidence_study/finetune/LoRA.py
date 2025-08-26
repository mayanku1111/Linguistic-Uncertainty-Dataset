import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from ..models import LLM
from ..confidence_extraction_methods import LinguisticConfidenceEstimator, DecisivenessEstimator
from peft import LoraConfig, TaskType, get_peft_model
from omegaconf import DictConfig
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq
from typing import Dict, List


class LoRA():
    def __init__(self, fintune_cfg, base_qa_model_cfg) -> None:
        self.fintune_cfg = fintune_cfg
        self.lora_cfg = self.get_lora_cfg(self.fintune_cfg)
        self.training_cfg = self.get_training_cfg(self.fintune_cfg)
        self.confidence_mapper = self.get_confidence_mapper(
            self.fintune_cfg)

        self.base_qa_model_cfg = base_qa_model_cfg
        self.base_qa_model = self.get_base_qa_model(self.base_qa_model_cfg)

    def get_base_qa_model(self, qa_model_cfg):
        return LLM(qa_model_cfg)

    def get_confidence_mapper(self, confidence_extraction_method_cfg):
        if confidence_extraction_method_cfg.mapper_name == "self-trained":
            confidence_estimator = LinguisticConfidenceEstimator(
                confidence_extraction_method_cfg)
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
            confidence_estimator.reg_head.load_state_dict(
                torch.load(confidence_extraction_method_cfg.state_dict_path))
            return confidence_estimator.to(device)
        elif confidence_extraction_method_cfg.mapper_name == "decisiveness":
            return DecisivenessEstimator(confidence_extraction_method_cfg)
        else:
            raise ValueError(
                f"Invalid confidence extraction method: {confidence_extraction_method_cfg}")

    def get_lora_cfg(self, fintune_cfg):
        return LoraConfig(
            r=fintune_cfg.r,
            target_modules=fintune_cfg.target_modules,
            lora_alpha=fintune_cfg.lora_alpha,
            lora_dropout=fintune_cfg.lora_dropout,
        )

    def get_training_cfg(self, fintune_cfg):
        return TrainingArguments(
            output_dir=fintune_cfg.output_dir,
            learning_rate=fintune_cfg.learning_rate,
            per_device_train_batch_size=fintune_cfg.per_device_train_batch_size,
            per_device_eval_batch_size=fintune_cfg.per_device_eval_batch_size,
            gradient_accumulation_steps=fintune_cfg.gradient_accumulation_steps,
            num_train_epochs=fintune_cfg.num_train_epochs,
            eval_strategy=fintune_cfg.eval_strategy,
            save_strategy=fintune_cfg.save_strategy,
            load_best_model_at_end=fintune_cfg.load_best_model_at_end,
            bf16=fintune_cfg.bf16,
            fp16=fintune_cfg.fp16,
        )

    def __call__(self, dataset, qa_batch_job_id, grader_batch_job_id) -> None:


        trainer = Trainer(
            model=lora_model,
            args=self.training_cfg,
            train_dataset=split_dataset["train"],
            eval_dataset=split_dataset["test"],
            processing_class=self.tokenizer,
            data_collator=self.data_collator,
        )
        trainer.train()
    
    def _tokenize(self, example: Dict) -> Dict[str, List[int]]:
        answer_text = str(example['answer']) if example['answer'] is not None else ""
        target_text = str(example['target_answer']) if example['target_answer'] is not None else ""

        prompt_text = f"User: {answer_text}\nAssistant: "

        prompt_ids = self.tokenizer.encode(
            prompt_text,
            add_special_tokens=True,
            padding=False,       
            truncation=True, 
            max_length=1024    
        )
        target_ids = self.tokenizer.encode(
            target_text,
            add_special_tokens=False,
            truncation=True,
            max_length=1024
        ) + [self.tokenizer.eos_token_id]

        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }