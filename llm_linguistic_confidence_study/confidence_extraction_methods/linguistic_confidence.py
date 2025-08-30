
import requests
import os
import pandas as pd
import re
from ..models import LLM
from omegaconf import DictConfig

SIMPLE_QA_EVAL_VANILLA_TEMPLATE = """
Answer the following question using a succinct (at most one sentence) and full answer.

Question: {question}
Answer:
""".strip()

SIMPLE_QA_EVAL_VANILLA_UNCERTAINTY_TEMPLATE = """
Answer the following question using a succinct (at most one sentence) and full answer. If you are uncertain about your answer to the question, convey this uncertainty linguistically by precisely hedging this answer.

Question: {question}
Answer:
""".strip()

DECISIVENESS_PROMPT = """You will be given a question and a proposed answer. Your task is to extract any assertions made by the proposed answer; for each assertion, you should output a
decisiveness score: a number in [0,1] representing how likely you think this assertion is to be true, judged solely based on the provided answer. You should pay
special attention to the usage of any hedging modifiers in the original answer, used to convey uncertainty in the truthfulness of the assertion. If the proposed
answer punts the question, the extracted assertion should be the empty string and the decisiveness score should be 1.0.

Here are some examples:

Question: criminal minds episode where jj becomes a profiler
Proposed answer: I think JJ officially becomes a profiler in Criminal Minds season 6, episode 2.
Extracted assertion: JJ officially becomes a profiler in Criminal Minds season 6, episode 2.
Decisiveness score: 0.8.

Question: criminal minds episode where jj becomes a profiler
Proposed answer: JJ officially becomes a profiler in Criminal Minds season 6, in either the first or second episode.
Extracted assertion: JJ officially becomes a profiler in Criminal Minds season 6, episode 1.
Decisiveness score: 0.5.
Extracted assertion: JJ officially becomes a profiler in Criminal Minds season 6, episode 2.
Decisiveness score: 0.5.

Question: criminal minds episode where jj becomes a profiler
Proposed answer: I'm not really sure about this, but I think the episode in which JJ officially becomes a profiler in Criminal Minds may be episode 2 in season 6.
Extracted assertion: JJ officially becomes a profiler in Criminal Minds season 6, episode 2.
Decisiveness score: 0.6.

Question: criminal minds episode where jj becomes a profiler
Proposed answer: I don't know which episode you're referring to.
Extracted assertion:
Decisiveness score: 1.0

Question: {question}
Proposed answer: {response}
""".strip()



import requests
import os

class LinguisticConfidenceExtractor():
    def __init__(self, model_name="meta-llama/llama-3.1-8b-instruct", api_key=None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")

    def extract_lvu(self, questions):
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        results = []
        for q in questions:
            prompt = SIMPLE_QA_EVAL_VANILLA_UNCERTAINTY_TEMPLATE.format(question=q)
            data = {
                "model": self.model_name,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                answer = response.json()["choices"][0]["message"]["content"]
            else:
                answer = None
            results.append({"question": q, "LVU_answer": answer})
        return results

    def extract_decisiveness(self, lvu_results):
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        scores = []
        for item in lvu_results:
            question = item["question"]
            answer = item["LVU_answer"]
            prompt = DECISIVENESS_PROMPT.format(question=question, response=answer)
            data = {
                "model": self.model_name,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                resp_text = response.json()["choices"][0]["message"]["content"]
                match = re.search(r"Decisiveness score: (\d+\.\d+)", resp_text)
                if match:
                    score = float(match.group(1))
                else:
                    score = None
            else:
                score = None
            scores.append(score)
        return scores
    def __init__(self, cfg: DictConfig):
        self.decisiveness_prompt_template = DECISIVENESS_PROMPT
        self.model = LLM(cfg.model)

    def __call__(self, dataset_df: pd.DataFrame) -> list[float]:
        decisiveness_prompts = []
        for i in range(len(dataset_df)):
            decisiveness_prompt = self.decisiveness_prompt_template.format(question=dataset_df.iloc[i]["question"], response=dataset_df.iloc[i]["response"])
            decisiveness_prompts.append(decisiveness_prompt)
        decisiveness_responses = self.model(decisiveness_prompts, task_name="decisiveness")
        decisiveness_scores = self.extract_confidence_scores(decisiveness_responses)
        return decisiveness_scores
    
    def extract_confidence_scores(self, decisiveness_responses: list[str]) -> list[float]:
        # use regex to extract the confidence score from the response
        # decisiveness_responses: list of strings
        # return a list of confidence scores in [0, 1]
        confidence_scores = []
        for response in decisiveness_responses:
            confidence_scores.append(float(re.search(r"Decisiveness score: (\d+\.\d+)", response).group(1)))
        return confidence_scores
    
