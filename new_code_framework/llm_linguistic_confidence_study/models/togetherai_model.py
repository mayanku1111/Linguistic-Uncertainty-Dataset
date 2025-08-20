from openai import OpenAI
from omegaconf import DictConfig
import json
import os

OPENAI_SYSTEM_PROMPT = "You are a helpful assistant."

class GPT():
    def __init__(self, model_cfg: DictConfig):
        self.model_cfg = model_cfg
        self.model = self.prepare_model(model_cfg)

    def __call__(self, prompts: list[str], task_name: str) -> list[str]:
        batch_job_id = self.prepare_batch_task(prompts, task_name)
        while self.check_batch_job_status(batch_job_id) != "completed":
            time.sleep(1)
        responses = self.retrieve_batch_job_output(batch_job_id)
        return responses
    
    def prepare_batch_task(self, prompts: list[str], task_name: str) -> list[str]:
        #########################################################
        # prepare request for batch grading
        #########################################################
        tasks = []
        for idx, prompt in enumerate(prompts):
            task = {
                "custom_id": task_name + "_" + str(idx),             # Custom ID must be a string
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model_cfg.name,         # Use gpt-5-mini
                    "messages": [
                        {
                            "role": "system",
                            "content": OPENAI_SYSTEM_PROMPT
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                }
            }
            tasks.append(task)

        # Write the tasks to a JSONL file
        with open(f'batch_tasks/{task_name}.jsonl', 'w', encoding='utf-8') as f:
            for t in tasks:
                f.write(json.dumps(t, ensure_ascii=False) + '\n')
        
        
        #########################################################
        # send request to batch grading
        #########################################################
        # Initialize OpenAI client with API key from environment variable
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Upload the batch task file; purpose must be "batch"
        batch_file = client.files.create(
            file=open(f'batch_tasks/{task_name}.jsonl', 'rb'),
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