import argparse
import os

from utils.generate_answers import *


def main(args):
    if args.test:
        print("TEST MODE")
        print(os.environ["HF_HOME"])
        input_path = args.input
        prompt_path = args.prompt
        output_path = os.path.join(os.getcwd(), r"src/linguistic_uncertainty_dataset/build/output",
                                   os.path.splitext(os.path.basename(input_path))[0] + "_answer.csv")
        if args.model == "Qwen/Qwen3-4B":
            generator = Generator_Qwen3_4B(
                model_id=args.model,
                prompt_template_path=prompt_path,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                MinP=0,
                enable_thinking=True,
                num_answers=10,
                device=args.cuda
            )
            output_path_4B = os.path.join(os.path.dirname(output_path),
                                          os.path.splitext(os.path.basename(output_path))[0] + "_4B" + os.path.splitext(output_path)[1])
            generator.generate(
                input_csv=input_path,
                output_csv=output_path_4B
            )
        elif args.model == "Qwen/Qwen3-0.6B":
            generator = Generator_Qwen3_0_6B(
                model_id=args.model,
                prompt_template_path=prompt_path,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                MinP=0,
                enable_thinking=True,
                num_answers=10
            )
            output_path_0_6B = os.path.join(os.path.dirname(output_path),
                                            os.path.splitext(os.path.basename(output_path))[0] + "_0_6B" + os.path.splitext(output_path)[1])
            generator.generate(
                input_csv=input_path,
                output_csv=output_path_0_6B
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
    argparser.add_argument("-input_csv", "--input_csv", "-input", "--input",
                           type=str,
                           dest="input",
                           default=os.path.join(
                               os.getcwd(), r"src/linguistic_uncertainty_dataset/res/questionlist_tmp_0_5.csv"),
                           help="input csv path")
    argparser.add_argument("-input_prompt", "--input_prompt", "-prompt", "--prompt",
                           type=str,
                           dest="prompt",
                           default=os.path.join(
                               os.getcwd(), r"src/linguistic_uncertainty_dataset/prompts/generate_answer.txt"),
                           help="prompt path")
    argparser.add_argument("-cuda", "--cuda",
                           type=str,
                           dest="cuda",
                           default="cuda:0",
                           help="GPU")
    args = argparser.parse_args()
    main(args)
