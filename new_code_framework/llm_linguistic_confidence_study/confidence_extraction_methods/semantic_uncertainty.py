import pandas as pd
import numpy as np
from sympy.logic.boolalg import false
import tqdm
from models import *


class SemanticUncertaintyExtractor():
    def __init__(self, config):
        self.dataset = config["dataset"]
        self.model_name = config["model_name"]

    def __call__(self, dataset: pd.DataFrame, model, sample_times: int = 10):
        df = self.generate_responses(model, dataset, sample_times)
        df = self.post_process_responses(df, sample_times)
        return df    # return a dataframe with the following columns: question, gold_answer, reponse1, reponse2, reponse3, ..., confidence, accuracy

    def prepare_prompt_sample(self):
        return """Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nYou will receive a question. Please provide an answer in a single brief but complete sentence.\n\n### Question:\n{question}\n\n### Response:"""

    def prepare_prompt_equivalence(self):
        return """We are evaluating answers to the question {question}\nHere are two possible answers:\nPossible Answer 1: {answer1}\nPossible Answer 2: {answer2}\nDoes Possible Answer 1 semantically entail Possible Answer 2? Respond only with entailment, contradiction, or neutral.\nResponse:"""

    def prepare_prompt_metric(self):
        return """The question is: {question}.\nThe correct answer to this question is: {correct_answer}.\nThe proposed answer is: {predicted_answer}\nWithin the context of the question, does the proposed answer mean the same as any of the expected answers? Respond only with yes or no.\nResponse:"""

    def generate_responses(self, model, df: pd.DataFrame, sample_times: int = 10):
        prompt_sample = self.prepare_prompt_sample()
        for i in range(sample_times):
            df[f"response{i+1}"] = None
        for idx, row in df.iterrows():
            responses = model.generate(prompt_sample.format(question=row["question"]))
            for i in range(sample_times):
                df.at[idx, f"response{i+1}"] = responses[i]
            df[f"response{idx+1}"] = responses[idx]["content"]
            df[f"prob{idx+1}"] = responses[idx]["logits"]
        return df

    # def generate_single_response(self, texts, model_name, sample_times):
    #     if model_name == "Qwen3/Qwen3-8B" or True:
    #         sampling_params = SamplingParams(
    #             temperature=0.6,
    #             top_p=0.95,
    #             top_k=20,
    #             max_tokens=32768,
    #             min_p=0.0,
    #             n=sample_times,
    #             logprobs=True,
    #         )
    #         outputs = self.llm.generate(texts, sampling_params)
    #         results = []
    #         for gen_out in outputs[0].outputs:
    #             output_ids = gen_out.token_ids
    #             logits = getattr(gen_out, "logprobs", None)
    #             try:
    #                 index = len(output_ids) - output_ids[::-1].index(151668)
    #             except ValueError:
    #                 index = 0
    #             thinking_content = self.tokenizer.decode(
    #                 output_ids[:index], skip_special_tokens=True
    #             ).strip("\n")
    #             content = self.tokenizer.decode(
    #                 output_ids[index:], skip_special_tokens=True
    #             ).strip("\n")
    #             results.append({
    #                 "thinking_content": thinking_content,
    #                 "split_index": index,
    #                 "content": content,
    #                 "gen_len": len(output_ids) - index,
    #                 "total_len": len(output_ids),
    #                 "logits": logits,
    #             })
    #         return results

    def post_process_responses(self, model, df: pd.DataFrame, sample_times: int = 10):
        # cluster
        prompt_equivalence = self.prepare_prompt_equivalence()
        df[f"entropy"] = None
        for i in range(sample_times):
            df[f"cluster{i+1}"] = None
            df[f"prob{i+1}"] = None
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Checking equivalence"):
            clusters = {}
            responses = {}
            total_clusters = 0
            for i in range(sample_times):
                now_response = df.at[idx, f"response{i+1}"]
                if now_response in responses:
                    df.at[idx, f"cluster{i+1}"] = responses[now_response]
                    clusters[responses[now_response]].append(i)
                    continue

                question = df.at[idx, f"question"]
                flag = True
                for j in range(j):
                    response = df.at[idx, f"response{j}"]
                    prompt_equivalence1 = prompt_equivalence.format(
                        question=question, answer1=now_response, answer2=response)
                    prompt_equivalence2 = prompt_equivalence.format(
                        question=question, answer2=now_response, answer1=response)
                    implication_1 = self.__check_(model, prompt_equivalence1)
                    implication_2 = self.__check_(model, prompt_equivalence2)
                    implications = [implication_1, implication_2]
                    semantically_equivalent = (0 not in implications) and (
                        [1, 1] != implications
                    )
                    if semantically_equivalent == 2:
                        df.at[idx, f"cluster{i+1}"] = responses[response]
                        flag = False
                        clusters[responses[response]].append(i)
                        break
                if flag:
                    total_clusters += 1
                    df.at[idx, f"cluster{i+1}"] = total_clusters
                    responses[now_response] = total_clusters
                    clusters[responses[response]] = [i]
            semantic_ids = []
            flag = True
            for i in range(sample_times):
                semantic_ids.append(df.at[idx, f"cluster{i+1}"])
                prompt_metric = self.prepare_prompt_metric().format(
                    question=df.at[idx, "question"],
                    correct_answer=df.at[idx, "gold_answer"],
                    predicted_answer=df.at[idx, f"response{i+1}"]
                )
                metric_answer = model.generate(prompt_metric)
                if 'yes' in metric_answer.lower() and flag:
                    df.at[idx, "response"] = df.at[idx, f"response{i}"]
                    flag = False

            # compute semantic entropy
            n_generations = len(semantic_ids)
            counts = np.bincount(semantic_ids)
            probabilities = counts/n_generations
            assert np.isclose(probabilities.sum(), 1)
            df.at[idx, "confidence"] = - \
                (probabilities * np.log(probabilities)).sum()
            df.at[idx, f"response"] = df.at[idx, f"response"]

        # return
        columns = ["question", "gold_answer"]
        for i in range(sample_times):
            columns.append(f"reponse{i+1}")
        columns.append("confidence")
        columns.append("accuracy")
        return df[columns]

    def calculate_confidence(self, model, question, responses, clusters, gold_answer):
        pass

    def calculate_accuracy(self, model, question, responses, clusters, gold_answer):
        pass

    def __check_(self, model, prompt):
        response = model.generate_answer(prompt)[0]
        if "entailment" in response:
            return 2
        elif "neutral" in response:
            return 1
        elif "contradiction" in response:
            return 0
        else:
            return 1
