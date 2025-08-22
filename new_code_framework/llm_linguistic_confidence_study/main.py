import pandas as pd
from confidence_extraction_methods import ConfidenceExtractor
from datasets import load_dataset
from metrics import MetricEvaluator
from omegaconf import OmegaConf, DictConfig
import hydra
import pandas as pd
from datetime import datetime
import logging


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    logging.info(OmegaConf.to_yaml(cfg, resolve=True))
    
    # load the dataset
    dataset = load_dataset(cfg.dataset)


    confidence_extractor = ConfidenceExtractor(cfg.confidence_extractor, cfg.qa_model)
    # return a dataframe with the following columns: question, gold_answer, reponse1, reponse2, reponse3, ..., confidence, accuracy
    responses_df = confidence_extractor(dataset, cfg.qa_batch_id, cfg.grader_batch_id)

    # evaluate the responses
    for metric_name in cfg.metrics:
        metric_evaluator = MetricEvaluator(metric_name)
        score = metric_evaluator.evaluate(responses_df)
        logging.info(f"{metric_name}: {score}")
        
    # save the results, responses and config
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    responses_df.to_csv(f"results/{cfg.dataset.name}/{cfg.confidence_extractor.name}/{cfg.qa_model.name}_{time_stamp}_responses.csv", index=False)
    OmegaConf.save(cfg, f"results/{cfg.dataset.name}/{cfg.confidence_extractor.name}/{cfg.qa_model.name}_{time_stamp}_config.yaml")
    logging.info(f"Responses saved to results/{cfg.dataset.name}/{cfg.confidence_extractor.name}/{cfg.qa_model.name}_{time_stamp}_responses.csv")
    logging.info(f"Config saved to results/{cfg.dataset.name}/{cfg.confidence_extractor.name}/{cfg.qa_model.name}_{time_stamp}_config.yaml")

if __name__ == "__main__":
    main()