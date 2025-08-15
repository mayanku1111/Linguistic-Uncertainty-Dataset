from simpleqa_eval import prepare_and_submit_simple_qa_eval_batch, check_batch_job_status, retrieve_batch_job_output
from confidence_mapper import UncertaintyRegressor
import torch
from transformers import AutoTokenizer, AutoModel
import argparse
import pandas as pd
from metrics import ECE
import os



# example script: python llm_uncertainty_eval.py --response_batch_job_id batch_689f470723288190a79fb84227cce093 --grader_batch_job_id batch_689f4f41fe3c8190beb3763dd2c0b142 --confidence_method vanilla_uncertainty
# example script: python llm_uncertainty_eval.py --response_batch_job_id batch_689da7e30b008190a9768c88d235d896 --grader_batch_job_id batch_689f4f03e654819081f9525271526e3a --confidence_method verbal_numerical_confidence
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--response_batch_job_id", type=str, required=True)
    parser.add_argument("--grader_batch_job_id", type=str, required=True)
    parser.add_argument("--confidence_method", type=str, default="vanilla", choices=["vanilla", "vanilla_uncertainty", "verbal_numerical_confidence"])
    args = parser.parse_args()

    # === initialize model ===
    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
    model = UncertaintyRegressor()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load("../tmp/uncertainty_model_human_anno_valid_count_2and1.pth", weights_only=True))

    # load the simple-qa dataset
    simple_qa_df = pd.read_csv("simple_qa_test_set.csv")
    # retrieve the simple-qa responses from the response batch job
    simple_qa_responses = retrieve_batch_job_output(args.response_batch_job_id, args.confidence_method)
    # retrieve the grader results from the grader batch job
    grader_results = retrieve_batch_job_output(args.grader_batch_job_id)


    if os.path.exists("llm_uncertainty_results.csv"):
        llm_uncertainty_results = pd.read_csv("llm_uncertainty_results.csv")
    else:
        llm_uncertainty_results = simple_qa_df.copy()
    llm_uncertainty_results[f"response_{args.confidence_method}"] = simple_qa_responses
    llm_uncertainty_results[f"grader_result_{args.confidence_method}"] = grader_results

    # === evaluate the confidence score using our linguistic confidence mapper ===
    inputs = tokenizer(simple_qa_responses, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.eval()
    with torch.no_grad():
        pred_scores = model(inputs["input_ids"], inputs["attention_mask"]).squeeze().cpu().numpy()
    if args.confidence_method in ["vanilla_uncertainty", "vanilla"]:
        llm_uncertainty_results[f"confidence_{args.confidence_method}"] = pred_scores
    elif args.confidence_method == "verbal_numerical_confidence":
        llm_uncertainty_results[f"confidence_{args.confidence_method}"] = pred_scores
        _, verbal_numerical_confidence_scores = retrieve_batch_job_output(args.response_batch_job_id, args.confidence_method, return_confidence_score=True)
        llm_uncertainty_results[f"confidence_{args.confidence_method}_verbal_numerical_confidence"] = [i/100 for i in verbal_numerical_confidence_scores]

    # === save the results ===
    llm_uncertainty_results.to_csv("llm_uncertainty_results.csv", index=False)
    print(llm_uncertainty_results.columns)