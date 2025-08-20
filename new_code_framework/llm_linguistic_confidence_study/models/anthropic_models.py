import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from omegaconf import DictConfig
import json
import os
import time

OPENAI_SYSTEM_PROMPT = "You are a helpful assistant."

class Claude():
    def __init__(self, model_cfg: DictConfig):
        self.model_cfg = model_cfg
        self.tasks = []

    def __call__(self, prompts: list[str], task_name: str) -> list[str]:
        batch_job_id = self.prepare_batch_task(prompts, task_name)
        while True:
            if self.check_batch_job_status(batch_job_id) != "ended":
                print("Batch ID: ", batch_job_id, self.check_batch_job_status(batch_job_id))
                break
            else:
                print("Batch ID: ", batch_job_id, self.check_batch_job_status(batch_job_id))
                time.sleep(60)
        responses = self.retrieve_batch_job_output(batch_job_id)
        return responses
    
    def prepare_batch_task(self, prompts: list[str], task_name: str):
        #########################################################
        # prepare request for batch grading
        #########################################################
        tasks = []
        for idx, prompt in enumerate(prompts):
            task = Request(
                custom_id = task_name + "_" + str(idx),      
                params = MessageCreateParamsNonStreaming(
                    model = self.model_cfg.name,  
                    messages = [
                        {
                            "role": "system",
                            "content": OPENAI_SYSTEM_PROMPT
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    system=[{"cache_control": {"type": "ephemeral"}}]
                )
            )
            tasks.append(task)
        self.tasks = tasks
        # Write the tasks to a JSONL file
        with open(f'batch_tasks/{task_name}.jsonl', 'w', encoding='utf-8') as f:
            for t in tasks:
                f.write(json.dumps(t, ensure_ascii=False) + '\n')
        
        
        #########################################################
        # send msg batch
        #########################################################
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        message_batch = client.messages.batches.create(tasks)        

        # return batch id
        return message_batch.id
    
    '''
    This function checks the status of a batch job.
    batch_job_id: the id of the batch job
    return: the status of the batch job
    '''
    def check_batch_job_status(self, batch_job_id):
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        message_batch = client.messages.batches.retrieve(batch_job_id)
        return message_batch.processing_status


    '''
    This function retrieves the output file of a batch job.
    batch_job_id: the id of the batch job
    confidence_method: the method of confidence score, if None, return the raw responses
    return_confidence_score: if True, return the confidence score of the responses (only for verbal_numerical_confidence method)
    return: a list of strings, each string is a response of a simple-qa question, in the same order as the simple-qa questions
    '''
    def retrieve_batch_job_output(self, batch_job_id):
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        responses = []
        results = list(client.messages.batches.results(batch_job_id))
        # Sort results by custom_id
        results.sort(key=lambda r: r.custom_id)

        for result in results:
            match result.result.type:
                case "succeeded":
                    print(f"Success! {result.custom_id}")
                    responses.append(result.result.message.content[0].text)
                case "errored":
                    responses.append(None)
                    if result.result.error.type == "invalid_request":
                        print(f"Validation error {result.custom_id}")
                    else:
                        print(f"Server error {result.custom_id}")
                case "expired":
                    responses.append(None)
                    print(f"Request expired {result.custom_id}")
                case _ :
                    responses.append(None)

        return responses


        
        
        