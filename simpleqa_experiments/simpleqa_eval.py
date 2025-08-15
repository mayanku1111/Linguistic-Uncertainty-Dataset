import pandas as pd
import json
from openai import OpenAI
import os
import argparse
import re

from prompt_templates import OPENAI_SYSTEM_PROMPT # system template
from prompt_templates import SIMPLE_QA_GRADER_TEMPLATE # grader template
from prompt_templates import SIMPLE_QA_EVAL_VANILLA_TEMPLATE, SIMPLE_QA_EVAL_VANILLA_UNCERTAINTY_TEMPLATE, SIMPLE_QA_EVAL_VERBAL_NUMERICAL_CONFIDENCE_TEMPLATE # eval template

eval_templates = {
    "vanilla": SIMPLE_QA_EVAL_VANILLA_TEMPLATE,
    "vanilla_uncertainty": SIMPLE_QA_EVAL_VANILLA_UNCERTAINTY_TEMPLATE,
    "verbal_numerical_confidence": SIMPLE_QA_EVAL_VERBAL_NUMERICAL_CONFIDENCE_TEMPLATE
}



'''
This function prepares a batch of simple-qa questions and answers for evaluation, the batch evaluation is dedicated to OpenAI batch evaluation API.
check more at https://platform.openai.com/docs/guides/batch#page-top

simple_qa_df: pandas dataframe, each row is a simple-qa question and gold answer
eval_model: the name of the model to use for evaluation
return: the batch job id
'''
def prepare_and_submit_simple_qa_eval_batch(simple_qa_df, eval_model, eval_template=SIMPLE_QA_EVAL_VANILLA_TEMPLATE):
    #########################################################
    # prepare request for batch evaluation
    #########################################################
    simple_qa_test_set_filename = 'simple_qa_test_set.csv'
    simple_qa_df = pd.read_csv(simple_qa_test_set_filename)

    tasks = []
    for idx, row in simple_qa_df.iterrows():
        task = {
            "custom_id": str(idx),             # Custom ID must be a string
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": eval_model,         # Use gpt-5-mini
                "messages": [
                    {
                        "role": "system",
                        "content": OPENAI_SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": eval_template.format(question=row["problem"])   # The question content
                    }
                ]
            }
        }
        tasks.append(task)

    # Write the tasks to a JSONL file
    with open(f'batch_tasks/simple_qa_batch_eval_{eval_model}_{args.confidence_method}.jsonl', 'w', encoding='utf-8') as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + '\n')
    
    #########################################################
    # send request to batch evaluation
    #########################################################
    # Initialize OpenAI client with API key from environment variable
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Upload the batch task file; purpose must be "batch"
    batch_file = client.files.create(
        file=open(f'batch_tasks/simple_qa_batch_eval_{eval_model}_{args.confidence_method}.jsonl', 'rb'),
        purpose='batch'
    )

    # Create the batch job; completion window can be 24h, 48h, etc.
    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    # return batch job id
    return batch_job.id


'''
This function checks the status of a batch job.
batch_job_id: the id of the batch job
return: the status of the batch job
'''
def check_batch_job_status(batch_job_id):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    batch_job = client.batches.retrieve(batch_job_id)
    return batch_job.status


'''
This function retrieves the output file of a batch job.
batch_job_id: the id of the batch job
confidence_method: the method of confidence score, if None, return the raw responses
return_confidence_score: if True, return the confidence score of the responses (only for verbal_numerical_confidence method)
return: a list of strings, each string is a response of a simple-qa question, in the same order as the simple-qa questions
'''
def retrieve_batch_job_output(batch_job_id, confidence_method=None, return_confidence_score=False):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    batch_job = client.batches.retrieve(batch_job_id)
    result_bytes = client.files.content(batch_job.output_file_id).content
    result_text  = result_bytes.decode('utf-8')

    results = []
    for line in result_text.strip().split("\n"):
        entry = json.loads(line)
        answer = entry["response"]["body"]["choices"][0]["message"]["content"].strip()
        results.append(answer)
    
    if confidence_method == "verbal_numerical_confidence":
        post_processed_results = []
        confidence_scores = []
        for response in results:
            fixed = re.sub(r'(?<=named the )"(.*)"(?=\.)', r'"\1"', response)
            fixed = fixed.replace('"Dulcie September Boardroom"', '\\"Dulcie September Boardroom\\"') # TODO: remove this
            data = json.loads(fixed)
            post_processed_results.append(data["answer"])
            confidence_scores.append(data["confidence_score"])
        if return_confidence_score:
            return post_processed_results, confidence_scores
        else:
            return post_processed_results
    else:
        return results


'''
This function submits a batch of simple-qa questions and answers to the grading API, defaulting to gpt-5-mini.

simple_qa_df: pandas dataframe, each row is a simple-qa question and gold answer
simple_qa_responses: list of strings, each string is a response of a simple-qa question, in the same order as the simple-qa questions
grader_model_name: the name of the model to use for grading
return: the batch job id
'''   
def prepare_and_submit_simple_qa_grader_batch(simple_qa_df, simple_qa_responses, eval_model, grader_model_name="gpt-5-mini", grader_template=SIMPLE_QA_GRADER_TEMPLATE):
    #########################################################
    # prepare request for batch grading
    #########################################################
    tasks = []
    for idx, row in simple_qa_df.iterrows():
        task = {
            "custom_id": str(idx),             # Custom ID must be a string
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": grader_model_name,         # Use gpt-5-mini
                "messages": [
                    {
                        "role": "system",
                        "content": OPENAI_SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": grader_template.format(question=row["problem"], target=row["answer"], predicted_answer=simple_qa_responses[idx])   # The question content
                    }
                ]
            }
        }
        tasks.append(task)

    # Write the tasks to a JSONL file
    with open(f'batch_tasks/simple_qa_batch_grader_{eval_model}.jsonl', 'w', encoding='utf-8') as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + '\n')
    
    
    #########################################################
    # send request to batch grading
    #########################################################
    # Initialize OpenAI client with API key from environment variable
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Upload the batch task file; purpose must be "batch"
    batch_file = client.files.create(
        file=open(f'batch_tasks/simple_qa_batch_grader_{eval_model}.jsonl', 'rb'),
        purpose='batch'
    )

    # Create the batch job; completion window can be 24h, 48h, etc.
    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    # return batch job id
    return batch_job.id




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["submit_eval", "submit_grader", "retrieve_grader_results"])
    parser.add_argument("--eval_model", type=str, required=True)
    parser.add_argument("--grader_model", type=str, default="gpt-5-mini")
    parser.add_argument("--grader_template", type=str, default=SIMPLE_QA_GRADER_TEMPLATE)
    parser.add_argument("--confidence_method", type=str, default="vanilla", choices=["vanilla", "vanilla_uncertainty", "verbal_numerical_confidence"])
    parser.add_argument("--response_batch_job_id", type=str, default=None)
    parser.add_argument("--grader_batch_job_id", type=str, default=None)
    parser.add_argument("--simple_qa_test_set_filepath", type=str, default="simple_qa_test_set.csv")
    args = parser.parse_args()
    
    # load the simple-qa test set
    simple_qa_df = pd.read_csv(args.simple_qa_test_set_filepath)

    if args.mode == "submit_eval":
        # submit eval batch job
        # example script: python simpleqa_eval.py --mode submit_eval --eval_model gpt-5-mini --eval_template vanilla_uncertainty
        batch_job_id = prepare_and_submit_simple_qa_eval_batch(simple_qa_df, args.eval_model, eval_templates[args.confidence_method])
        print(f"Eval batch job id: {batch_job_id}")

        
    elif args.mode == "submit_grader":
        #########################################################
        # 1. retrieve the responses
        # 2. prepare and submit the grader batch job
        # 
        # script to use: python simpleqa_eval.py --mode submit_grader --response_batch_job_id <response_batch_job_id> --eval_model <eval_model> --grader_model <grader_model> --grader_template <grader_template>
        # example: python simpleqa_eval.py --mode submit_grader --eval_model gpt-5-mini --response_batch_job_id batch_689eced6e8308190adaac64542a3dfbc --confidence_method vanilla
        #########################################################
        if args.response_batch_job_id is None:
            # TODO: other way to get the responses
            pass
        else:
            # retrieve the responses from the response batch job
            simple_qa_responses = retrieve_batch_job_output(args.response_batch_job_id, args.confidence_method)
        # 2. prepare and submit the grader batch job
        batch_job_id = prepare_and_submit_simple_qa_grader_batch(simple_qa_df, simple_qa_responses, args.eval_model, args.grader_model, args.grader_template)
        print(f"Grader batch job id: {batch_job_id}")
        
        
        
    elif args.mode == "retrieve_grader_results":
        #########################################################
        # 1. retrieve the grader results
        # 
        # script to use: python simpleqa_eval.py --mode retrieve_grader_results --eval_model <eval_model> --response_batch_job_id <response_batch_job_id> --grader_batch_job_id <grader_batch_job_id>
        # example: python simpleqa_eval.py --mode retrieve_grader_results --eval_model gpt-5-mini --response_batch_job_id batch_689da7dd35708190a6d0ed4012152e50 --grader_batch_job_id batch_689eedb57d9c81909f0bd660fa2a3271
        #########################################################
        if args.grader_batch_job_id is None:
            raise ValueError("grader_batch_job_id is required for retrieve_grader_results mode")
        else:
            # retrieve the responses from the response batch job
            # `simple_qa_responses` is a list of strings that contains the responses of simple-qa
            # questions. Each string in the list corresponds to the response of a specific simple-qa
            # question, and the order of the strings in the list matches the order of the questions in
            # the dataset. This list is used in the context of evaluating the responses generated by a
            # model for simple-qa questions.
            simple_qa_responses = retrieve_batch_job_output(args.response_batch_job_id)
            # retrieve the grader results from the grader batch job
            grader_results = retrieve_batch_job_output(args.grader_batch_job_id)
            
            As = Bs = Cs = 0
            for response, grader_result in zip(simple_qa_responses, grader_results):
                print(f"Response: {response}, grader result: {grader_result}")
                if grader_result == "A":
                    As += 1
                elif grader_result == "B":
                    Bs += 1
                elif grader_result == "C":
                    Cs += 1
                print("-"*100)
            print(f"As: {As}, Bs: {Bs}, Cs: {Cs}")
                
                
                
                
                
                
    else:
        raise ValueError(f"Invalid mode: {args.mode}")
    
    