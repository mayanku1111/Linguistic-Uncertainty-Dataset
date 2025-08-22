from .confidence_extraction_methods import ConfidenceExtractor
from .datasets import load_dataset
from .metrics import MetricEvaluator
from omegaconf import OmegaConf, DictConfig
import hydra
from datetime import datetime
import logging
import os
import shutil
from hydra.core.hydra_config import HydraConfig


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    logging.info(OmegaConf.to_yaml(cfg, resolve=True))
    
    # load the dataset
    dataset = load_dataset(cfg.dataset)
    

    confidence_extractor = ConfidenceExtractor(cfg.confidence_extractor, cfg.qa_model)
    # return a dataframe with the following columns: question, gold_answer, reponse1, reponse2, reponse3, ..., confidence, accuracy
    responses_df = confidence_extractor(dataset, cfg.qa_batch_id, cfg.grader_batch_id)

    # evaluate the responses
    for name, metric_cfg in cfg.metrics.items():
        metric_evaluator = MetricEvaluator(metric_cfg, dataset)
        score = metric_evaluator.evaluate(responses_df)
        logging.info(f"{metric_cfg}: {score}")
        
    # save the results, responses and config
    save_dir = f"llm_linguistic_confidence_study/results/{cfg.dataset.name}/{cfg.confidence_extractor.name}"
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(save_dir, exist_ok=True)
    responses_df.to_csv(os.path.join(save_dir, f"{cfg.qa_model.name}_{time_stamp}_responses.csv"), index=False)
    OmegaConf.save(cfg, os.path.join(save_dir, f"{cfg.qa_model.name}_{time_stamp}_config.yaml"))
    logging.info(f"Outputs saved to {save_dir}")
    
    # copy Hydra main.log into results folder
    output_dir = HydraConfig.get().runtime.output_dir
    hydra_log_path = os.path.join(output_dir, "main.log")
    dst_log_path = os.path.join(save_dir, f"{cfg.qa_model.name}_{time_stamp}_main.log")
    shutil.copy(hydra_log_path, dst_log_path)

if __name__ == "__main__":
    main()