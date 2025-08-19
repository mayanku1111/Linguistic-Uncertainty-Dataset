import pandas as pd
from confidence_extraction_methods import ConfidenceExtractor
from datasets import load_dataset
from metrics import MetricEvaluator
from omegaconf import OmegaConf, DictConfig
import hydra
import pandas as pd
import os
from datetime import datetime
import logging


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    logging.info(OmegaConf.to_yaml(cfg, resolve=True))
    
    # load the dataset
    qa_data = load_dataset(cfg.dataset)

    # obtain the responses
    if cfg.eval_only:
        responses_df = pd.read_csv(cfg.responses_path)
    else:
        confidence_extractor = ConfidenceExtractor(cfg.confidence_extraction_method, cfg.confidence_extraction_method_config, qa_data)
        responses_df = confidence_extractor.generate_responses()

    # evaluate the responses
    results_df = pd.DataFrame(columns=cfg.metrics_list)
    for metric in cfg.metrics_list:
        metric_evaluator = MetricEvaluator(metric)
        results_df[metric] = metric_evaluator.evaluate(responses_df)
        logging.info(f"{metric}: {results_df[metric]}")
        
    # save the results, responses and config
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    responses_df.to_csv(f"results/{cfg.dataset}/{cfg.confidence_extraction_method}/{cfg.model}_{time_stamp}_responses.csv", index=False)
    results_df.to_csv(f"results/{cfg.dataset}/{cfg.confidence_extraction_method}/{cfg.model}_{time_stamp}_results.csv", index=False)
    OmegaConf.save(cfg, f"results/{cfg.dataset}/{cfg.confidence_extraction_method}/{cfg.model}_{time_stamp}_config.yaml")
    logging.info(f"Results saved to results/{cfg.dataset}/{cfg.confidence_extraction_method}/{cfg.model}_{time_stamp}_results.csv")
    logging.info(f"Responses saved to results/{cfg.dataset}/{cfg.confidence_extraction_method}/{cfg.model}_{time_stamp}_responses.csv")
    logging.info(f"Config saved to results/{cfg.dataset}/{cfg.confidence_extraction_method}/{cfg.model}_{time_stamp}_config.yaml")

if __name__ == "__main__":
    main()