import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os
import abc


class Base_Generator(abc.ABC):
    def __init__(self):
        pass
    
    def generate(self, input_csv: str, output_csv: str) -> None:
        print("Generate answer start.")
        print(f"Input File Path: {input_csv}")
        print(f"Output File Path: {output_csv}")
        self.generate_from_csv(input_csv, output_csv)
        print(f"Generate answer end.")


    @abc.abstractmethod
    def generate_from_csv(self, input_csv: str, output_csv: str) -> None:
        raise NotImplementedError


class Generator_Qwen(Base_Generator):
    def __init__(self):
        return super().__init__()

    @abc.abstractmethod
    def _build_prompt(self, question: str) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def _generate_answer(self, prompt: str) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def _get_num_answers(self) -> int:
        raise NotImplementedError

    def generate_from_csv(self, input_csv: str, output_csv: str) -> None:
        if not os.path.exists(input_csv):
            raise FileNotFoundError(
                f"No such file or directory: '{input_csv}'")

        df = pd.read_csv(input_csv)
        if "problem" not in df.columns:
            raise ValueError("Missing column 'problem'.")

        done_set = set()
        if os.path.exists(output_csv):
            existing_df = pd.read_csv(output_csv)
            for _, row in existing_df.iterrows():
                done_set.add((row['problem'], row['answer_index']))

        write_header = not os.path.exists(output_csv)
        output_file = open(output_csv, "a", encoding="utf-8", newline='')
        import csv
        writer = csv.DictWriter(output_file, fieldnames=[
                                "problem", "answer_index", "answer"])
        if write_header:
            writer.writeheader()

        # 逐条生成并写入
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating answers"):
            question = row['problem']
            prompt = self._build_prompt(question)

            for i in range(1, self._get_num_answers() + 1):
                if (question, i) in done_set:
                    continue
                try:
                    answer = self._generate_answer(prompt)
                    writer.writerow({
                        "problem": question,
                        "answer_index": i,
                        "answer": answer
                    })
                    output_file.flush()
                    done_set.add((question, i))
                except Exception as e:
                    print(f"Error in generate answers", {e})
                    continue

        output_file.close()
        print(f"Answers output in {output_csv}")


class Generator_Qwen3_4B(Generator_Qwen):
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-4B",
        prompt_template_path: str = r"./prompts/generate_answer.txt",
        device: str = "cuda:1",
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        MinP=0,
        enable_thinking=True,
        num_answers=10,
        max_new_tokens=32768,
    ):
        self.model_id = model_id
        self.prompt_template_path = prompt_template_path
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self.MinP = 0
        self.num_answers = num_answers
        self.enable_thinking = enable_thinking
        if torch.cuda.is_available():
            self.device = device
        else:
            raise ("ERROR: Current GPU {self.device} not available.")

        self._load_model()
        self._load_prompt()

    def _load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True,)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype="auto", trust_remote_code=True,)
        self.model.to(self.device).eval()

    def _load_prompt(self):
        if not os.path.exists(self.prompt_template_path):
            raise FileNotFoundError(
                f"No such file or directory: '{self.prompt_template_path}'")
        with open(self.prompt_template_path, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()
        if "{question}" not in self.prompt_template:
            raise ValueError(f"Missing placeholder {{question}} in prompt.")

    def _build_prompt(self, question: str) -> str:
        return self.prompt_template.format(question=question)

    def _generate_answer(self, prompt: str) -> str:
        messages = [
            {"role": "user",
             "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.MinP,
        )
        # Denote this line to obtain full answer.
        # decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        output_ids = outputs[0][len(inputs.input_ids[0]):].tolist()
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        # Denote this line to obtain thinking content.
        # thinking_content = self.tokenizer.decoder(output_ids[:index], skip_special_tokens=True)

        # This line is answer content.
        content = self.tokenizer.decode(
            output_ids[index:], skip_special_tokens=True)

        return content.replace(prompt, "").strip()

    def _get_num_answers(self) -> int:
        return self.num_answers


class Generator_Qwen3_0_6B(Generator_Qwen):
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-0.6B",
        prompt_template_path: str = r"./prompts/generate_answer.txt",
        device: str = "cuda:1",
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        MinP=0,
        enable_thinking=True,
        num_answers=10,
        max_new_tokens=32768,
    ):
        self.model_id = model_id
        self.prompt_template_path = prompt_template_path
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self.MinP = 0
        self.num_answers = num_answers
        self.enable_thinking = enable_thinking
        if torch.cuda.is_available():
            self.device = device
        else:
            raise ("ERROR: Current GPU {self.device} not available.")

        self._load_model()
        self._load_prompt()

    def _load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True,)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype="auto", trust_remote_code=True,)
        self.model.to(self.device).eval()

    def _load_prompt(self):
        if not os.path.exists(self.prompt_template_path):
            raise FileNotFoundError(
                f"No such file or directory: '{self.prompt_template_path}'")
        with open(self.prompt_template_path, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()
        if "{question}" not in self.prompt_template:
            raise ValueError(f"Missing placeholder {{question}} in prompt.")

    def _build_prompt(self, question: str) -> str:
        return self.prompt_template.format(question=question)

    def _generate_answer(self, prompt: str) -> str:
        messages = [
            {"role": "user",
             "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.MinP,
        )
        # Denote this line to obtain full answer.
        # decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        output_ids = outputs[0][len(inputs.input_ids[0]):].tolist()
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        # Denote this line to obtain thinking content.
        # thinking_content = self.tokenizer.decoder(output_ids[:index], skip_special_tokens=True)

        # This line is answer content.
        content = self.tokenizer.decode(
            output_ids[index:], skip_special_tokens=True)

        return content.replace(prompt, "").strip()

    def _get_num_answers(self) -> int:
        return self.num_answers