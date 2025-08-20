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

    # obtain the responses
    if cfg.eval_only:       
        responses_df = pd.read_csv(cfg.responses_path)
    else:
        confidence_extractor = ConfidenceExtractor(cfg.confidence_extractor, cfg.qa_model)
        # return a dataframe with the following columns: question, gold_answer, reponse1, reponse2, reponse3, ..., confidence, accuracy
        responses_df = confidence_extractor(dataset)

    # evaluate the responses
    results_df = pd.DataFrame(columns=cfg.metrics)
    for metric in cfg.metrics:
        metric_evaluator = MetricEvaluator(metric)
        results_df[metric] = metric_evaluator.evaluate(responses_df)
        logging.info(f"{metric}: {results_df[metric]}")
        
    # save the results, responses and config
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    responses_df.to_csv(f"results/{cfg.dataset.name}/{cfg.confidence_extractor.name}/{cfg.qa_model.name}_{time_stamp}_responses.csv", index=False)
    results_df.to_csv(f"results/{cfg.dataset.name}/{cfg.confidence_extractor.name}/{cfg.qa_model.name}_{time_stamp}_results.csv", index=False)
    OmegaConf.save(cfg, f"results/{cfg.dataset.name}/{cfg.confidence_extractor.name}/{cfg.qa_model.name}_{time_stamp}_config.yaml")
    logging.info(f"Results saved to results/{cfg.dataset.name}/{cfg.confidence_extractor.name}/{cfg.qa_model.name}_{time_stamp}_results.csv")
    logging.info(f"Responses saved to results/{cfg.dataset.name}/{cfg.confidence_extractor.name}/{cfg.qa_model.name}_{time_stamp}_responses.csv")
    logging.info(f"Config saved to results/{cfg.dataset.name}/{cfg.confidence_extractor.name}/{cfg.qa_model.name}_{time_stamp}_config.yaml")

if __name__ == "__main__":
    main()