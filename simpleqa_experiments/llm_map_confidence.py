# write a script that ask gpt-4o-mini to map the uncertainty expression (a sentence) to a confidence score with the following format:
# Please provide a confidence score between 0 and 100, based solely on the degree of confidence expressed in the tone of the following sentence (without using any external or prior knowledge): {UNCERTAINTY EXPRESSION}

import openai
import pandas as pd
import os
from tqdm import tqdm
import argparse

openai.api_key = os.environ.get("OPENAI_API_KEY")

def map_confidence(sentence, model_name="gpt-4o-mini"):
    # Use the new openai>=1.0.0 API for chat completions
    response = openai.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": f"Please provide only a confidence score between 0 and 100, based solely on the degree of confidence expressed in the tone of the following sentence (without using any external or prior knowledge): {sentence}"
            }
        ]
    )
    return response.choices[0].message.content

def map_confidence_using_google_paper(question, uncertainty_expression, model_name="gpt-5-mini"):
    """
    Follow paper: Can Large Language Models Faithfully Express Their Intrinsic Uncertainty in Words?
    """
    PAPER_DECISIVENESS_PROMPT = """You will be given a question and a proposed answer. Your task is to extract any assertions made by the proposed answer; for each assertion, you should output a
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

    Question: {Question}
    Proposed answer: {Response}
    """.strip()
    # Use the new openai>=1.0.0 API for chat completions
    response = openai.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": PAPER_DECISIVENESS_PROMPT.format(Question=question, Response=uncertainty_expression)
            }
        ]
    )
    return response.choices[0].message.content



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt-5-mini")
    parser.add_argument("--use_google_paper", action="store_true", default=False)
    args = parser.parse_args()
    
    df = pd.read_csv("tmp/dataset_valid_confidence_score_count_3_eval.csv")
    for index, row in df.iterrows():
        sentence = row["uncertainty_expression"]
        if args.use_google_paper:
            confidence = map_confidence_using_google_paper(row["question"], sentence, model_name=args.model_name)
        else:
            confidence = map_confidence(sentence, model_name=args.model_name)
        df.at[index, f"confidence_score_{args.model_name}"] = confidence
        print(f"{args.model_name} {index+1}/{len(df)}: Sentence: {sentence}, Confidence: {confidence}")
    if args.use_google_paper:
        df.to_csv(f"tmp/dataset_valid_confidence_score_count_3_{args.model_name}_google_paper.csv", index=False)
        print(f"Saved to tmp/dataset_valid_confidence_score_count_3_{args.model_name}_google_paper.csv")
    else:
        df.to_csv(f"tmp/dataset_valid_confidence_score_count_3_{args.model_name}.csv", index=False)
        print(f"Saved to tmp/dataset_valid_confidence_score_count_3_{args.model_name}.csv")