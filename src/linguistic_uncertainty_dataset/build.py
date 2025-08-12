import argparse
import os

from utils.generate_answers import *


def main(args):
    if args.test:
        print("TEST MODE")
        print(os.environ["HF_HOME"])
        generator = Generator_Qwen3_4B(
            model_id=args.model,
            prompt_template_path=r"prompts/generate_answer.txt",
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            MinP=0,
            enable_thinking=True,
            num_answers=10
        )
        generator.generate_from_csv(
            input_csv="./res/questionlist_tmp_0_5.csv",
            output_csv="./build/output/questionlist_answer_tmp_0_5.csv"
        )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Uncertainty Finetuning Script")

    argparser.add_argument("-b", "-build", "--build", dest="build",
                           action="store_true",  help="Run in build mode",)
    argparser.add_argument("-t", "-test", "--t", "--test",
                           dest="test", action="store_true", help="test mode",)
    argparser.add_argument("-model", "-model_id", "--model_id", type=str,
                           dest="model", default="Qwen/Qwen3-4B", help="model_id")

    args = argparser.parse_args()

    main(args)
