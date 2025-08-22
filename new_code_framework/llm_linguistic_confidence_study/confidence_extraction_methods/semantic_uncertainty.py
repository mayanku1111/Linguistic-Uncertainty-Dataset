import pandas as pd
import numpy as np
from sympy.logic.boolalg import false
import tqdm
from models import *
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import os
import logging

# new version
class SemanticUncertaintyExtractor():
    def __init__(self, confidence_extraction_method_cfg, qa_model_cfg):
        self.confidence_extraction_method_cfg = confidence_extraction_method_cfg
        self.qa_model_cfg = qa_model_cfg
        self.qa_model = self.get_qa_model(self.qa_model_cfg)
        self.entailment_model = EntailmentDeberta()
        
    def get_qa_model(self, qa_model_cfg):
        return LLM(qa_model_cfg)
    
    def __call__(self, dataset, qa_batch_job_id: str = None, grader_batch_job_id: str = None):
        if dataset.name == "simple_qa" or dataset.name == "mini_simple_qa":
            qa_responses = self.generate_qa_responses(dataset.df, self.confidence_extraction_method_cfg, task_name=f"simple_qa_su_qa", qa_batch_job_id=qa_batch_job_id)
            # combine qa_responses and dataset_df
            response_df = dataset.df.copy()
            for i in range(self.confidence_extraction_method_cfg.sample_times):
                response_df[f"response_{i}"] = qa_responses[i]
            # semantic uncertainty estimation
            semantic_ids = self.get_semantic_ids(response_df, self.entailment_model)
            for i in range(self.confidence_extraction_method_cfg.sample_times):
                response_df[f"semantic_id_{i}"] = semantic_ids[i]
            # confidence estimation
            confidences = self.calculate_confidence(response_df)
            response_df["confidences"] = confidences
            # grade the accuracy of the confidence scores
            accuracies = dataset.grade_responses(response_df["responses"], grader_batch_job_id=grader_batch_job_id)
            response_df["accuracies"] = accuracies
        elif dataset.name == "mmlu_pro":
            pass
        else:
            raise ValueError(f"Invalid dataset name: {dataset.name}")
        # return the response_df
        return response_df

    def get_semantic_ids(self, response_df: pd.DataFrame, model, strict_entailment=False, example=None):
        """Group list of predictions into semantic meaning."""

        def are_equivalent(text1, text2):

            implication_1 = model.check_implication(text1, text2, example=example)
            implication_2 = model.check_implication(text2, text1, example=example)  # pylint: disable=arguments-out-of-order
            assert (implication_1 in [0, 1, 2]) and (implication_2 in [0, 1, 2])

            if strict_entailment:
                semantically_equivalent = (implication_1 == 2) and (implication_2 == 2)

            else:
                implications = [implication_1, implication_2]
                # Check if none of the implications are 0 (contradiction) and not both of them are neutral.
                semantically_equivalent = (0 not in implications) and ([1, 1] != implications)

            return semantically_equivalent

        # Initialise all ids with -1.
        semantic_set_ids = [-1] * len(strings_list)
        # Keep track of current id.
        next_id = 0
        for i, string1 in enumerate(strings_list):
            # Check if string1 already has an id assigned.
            if semantic_set_ids[i] == -1:
                # If string1 has not been assigned an id, assign it next_id.
                semantic_set_ids[i] = next_id
                for j in range(i+1, len(strings_list)):
                    # Search through all remaining strings. If they are equivalent to string1, assign them the same id.
                    if are_equivalent(string1, strings_list[j]):
                        semantic_set_ids[j] = next_id
                next_id += 1

        assert -1 not in semantic_set_ids

        return semantic_set_ids


class EntailmentDeberta():
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xlarge-mnli")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v2-xlarge-mnli").to(device)

    def check_implication(self, text1, text2, *args, **kwargs):
        inputs = self.tokenizer(text1, text2, return_tensors="pt").to(device)
        # The model checks if text1 -> text2, i.e. if text2 follows from text1.
        # check_implication('The weather is good', 'The weather is good and I like you') --> 1
        # check_implication('The weather is good and I like you', 'The weather is good') --> 2
        outputs = self.model(**inputs)
        logits = outputs.logits
        # Deberta-mnli returns `neutral` and `entailment` classes at indices 1 and 2.
        largest_index = torch.argmax(F.softmax(logits, dim=1))  # pylint: disable=no-member
        prediction = largest_index.cpu().item()
        if os.environ.get('DEBERTA_FULL_LOG', False):
            logging.info('Deberta Input: %s -> %s', text1, text2)
            logging.info('Deberta Prediction: %s', prediction)

        return prediction


# old version
class SemanticUncertaintyExtractor_old():
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
