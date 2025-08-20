from together import Together
from omegaconf import DictConfig
import json
import os
import time

OPENAI_SYSTEM_PROMPT = "You are a helpful assistant."

class TogetherAI():
    def __init__(self, model_cfg: DictConfig):
        self.model_cfg = model_cfg
        self.tasks = []

    def __call__(self, prompts: list[str], task_name: str) -> list[str]:
        batch_job_id = self.prepare_batch_task(prompts, task_name)
        while True:
            if self.check_batch_job_status(batch_job_id) != "COMPLETED":
                print("Batch ID: ", batch_job_id, self.check_batch_job_status(batch_job_id))
                break
            else:
                print("Batch ID: ", batch_job_id, self.check_batch_job_status(batch_job_id))
                time.sleep(60)
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
        self.tasks = tasks
        # Write the tasks to a JSONL file
        with open(f'batch_tasks/{task_name}.jsonl', 'w', encoding='utf-8') as f:
            for t in tasks:
                f.write(json.dumps(t, ensure_ascii=False) + '\n')
        
        
        #########################################################
        # send request to batch grading
        #########################################################
        # Initialize OpenAI client with API key from environment variable
        client = Together(api_key=os.getenv("TOGETHERAI_API_KEY"))

        # Upload the batch task file; purpose must be "batch"
        file_resp = client.files.upload(
            file=open(f'batch_tasks/{task_name}.jsonl', 'rb'),
            purpose='batch'
        )
        file_id = file_resp.id

        # Create the batch job 
        batch = client.batches.create_batch(file_id, endpoint="/v1/chat/completions")

        # return batch id
        return batch.id
    
    '''
    This function checks the status of a batch job.
    batch_job_id: the id of the batch job
    return: the status of the batch job
    '''
    def check_batch_job_status(self, batch_job_id):
        client = Together(api_key=os.getenv("TOGETHERAI_API_KEY"))
        batch_job = client.batches.get_batch(batch_job_id)
        return batch_job.status


    '''
    This function retrieves the output file of a batch job.
    batch_job_id: the id of the batch job
    confidence_method: the method of confidence score, if None, return the raw responses
    return_confidence_score: if True, return the confidence score of the responses (only for verbal_numerical_confidence method)
    return: a list of strings, each string is a response of a simple-qa question, in the same order as the simple-qa questions
    '''
    def retrieve_batch_job_output(self, batch_job_id):
        client = Together(api_key=os.getenv("TOGETHERAI_API_KEY"))
        file_path = f"batch_tasks/tmp_together_{batch_job_id}_batch_output.jsonl"
        batch_stat = client.batches.get_batch(batch_job_id)
        if self.check_batch_job_status(batch_job_id) == 'COMPLETED':
            client.files.retrieve_content(id=batch_stat.output_file_id, output=file_path)
        responses = []
        with open(file_path, "r") as f:
            for line in f:
                if line.strip():  # skip empty lines
                    responses.append(json.loads(line))
        
        ordered_responses = []
        responded_ids = [r["custom_id"] for r in responses]
        for task in self.tasks:
            id = task["custom_id"]
            if id in responded_ids:
                for r in responses:
                    if id == r["custom_id"]:
                        ordered_responses.append(r["response"]["body"]["choices"][0]["message"]["content"].strip())
            else:
                ordered_responses.append(None)
                
        return ordered_responses



        
        
        