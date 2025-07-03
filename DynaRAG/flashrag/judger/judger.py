from typing import cast, List
import json
from collections import Counter
import numpy as np
import torch
import faiss
from flashrag.retriever.utils import load_model, pooling

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType, PeftModel
from datasets import Dataset


class BaseJudger:
    """Base object of Judger, used for judging whether to retrieve"""

    def __init__(self, config):
        self.config = config
        self.name = config['judger_name']
        self.device = config['device']
        print(f"[BaseJudger] Initialized with name: {self.name}, device: {self.device}")

    def run(self, item) -> str:
        """Get judgement result.
        Args:
            item: dataset item, contains question, retrieval result...

        Returns:
            judgement: bool, whether to retreive
        """
        pass

    def batch_run(self, dataset, batch_size=None) -> List[str]:
        print("[BaseJudger] Running batch...")
        return [self.run(item) for item in dataset]


class SKRJudger(BaseJudger):
    """Implementation for SKR-knn
    Paper link: https://aclanthology.org/2023.findings-emnlp.691.pdf
    """

    def __init__(self, config):
        super().__init__(config)
        self.model_path = config['judger_model_path']
        self.training_data_path = config['judger_training_data_path']
        print(f"[SKRJudger] Loading model from {self.model_path}")
        self.encoder, self.tokenizer = load_model(model_path=self.model_path, use_fp16=False)
        self.topk = config['judger_topk'] if 'judger_topk' in config else 5
        self.batch_size = config['judger_batch_size'] if 'judger_batch_size' in config else 64
        self.max_length = config['judger_max_length'] if 'judger_max_length' in config else 128

        print(f"[SKRJudger] Loading training data from {self.training_data_path}")
        with open(self.training_data_path, "r") as f:
            self.training_data = json.load(f)

        print("[SKRJudger] Counting positive and negative samples in training data")
        self.training_data_counter = Counter([item['judgement'].strip() for item in self.training_data])
        self.training_pos_num = self.training_data_counter['ir_better']
        self.training_neg_num = self.training_data_counter['ir_worse']
        self.training_data_num = sum(self.training_data_counter.values())

        print("[SKRJudger] Encoding training questions into FAISS")
        training_questions = [item['question'] for item in self.training_data]
        all_embeddings = self.encode(training_questions)
        faiss_index = faiss.index_factory(all_embeddings.shape[-1], 'Flat', faiss.METRIC_L2)
        faiss_index.add(all_embeddings)
        self.faiss = faiss_index

    @torch.inference_mode(mode=True)
    def encode(self, contents: list):
        print(f"[SKRJudger] Encoding {len(contents)} items")
        inputs = self.tokenizer(contents, padding=True, truncation=True, return_tensors='pt',
                                max_length=self.max_length).to('cuda')
        output = self.encoder(**inputs, return_dict=True)
        embeddings = pooling(output.pooler_output, output.last_hidden_state, inputs['attention_mask'], 'pooler')
        embeddings = cast(torch.Tensor, embeddings)
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1).detach()
        all_embeddings = embeddings.cpu().numpy().astype(np.float32)
        return all_embeddings

    def judge(self, dataset):
        print("[SKRJudger] Judging dataset")
        questions = dataset.question
        all_judgements = []
        for start_idx in range(0, len(questions), self.batch_size):
            batch_question = questions[start_idx:start_idx + self.batch_size]
            print(f"[SKRJudger] Processing batch {start_idx} to {start_idx + self.batch_size}")
            batch_emb = self.encode(batch_question)
            scores, batch_idxs = self.faiss.search(batch_emb, k=self.topk)

            for idxs in batch_idxs:
                topk_samples = [self.training_data[idx]['judgement'].strip() for idx in idxs]
                topk_counter = Counter(topk_samples)

                ir_better_num = topk_counter['ir_better']
                ir_worse_num = topk_counter['ir_worse']
                topk_delta = ir_better_num - ir_worse_num
                training_data_delta = self.training_pos_num - self.training_neg_num

                if training_data_delta < 0:
                    if topk_delta < 0 and topk_delta <= int(training_data_delta * self.topk / self.training_data_num):
                        judgement = False
                    else:
                        judgement = True
                else:
                    if topk_delta > 0 and topk_delta >= int(training_data_delta * self.topk / self.training_data_num):
                        judgement = True
                    else:
                        judgement = False

                all_judgements.append(judgement)
        print(f"[SKRJudger] Judgements: {all_judgements}")
        return all_judgements


class LoraJudger(BaseJudger):
    def __init__(self, config):
        super().__init__(config)
        mode_path = config['mode_path']
        lora_path = config['lora_path']

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1
        )

        print(f"[LoraJudger] Loading tokenizer from {mode_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(mode_path)

        print(f"[LoraJudger] Loading model from {mode_path}")
        model = AutoModelForCausalLM.from_pretrained(mode_path, device_map="auto", torch_dtype=torch.bfloat16)

        print(f"[LoraJudger] Loading LoRA weights from {lora_path}")
        self.model = PeftModel.from_pretrained(model, model_id=lora_path, config=lora_config)

    def infer(self, question: str) -> str:
        prompt = f"你可以独立回答这个问题吗？请只回答“True”或“False”：{question}"
        messages = [
            {"role": "system", "content": "现在你是一个判断器"},
            {"role": "user", "content": prompt}
        ]

        print(f"[LoraJudger] Inferring for question: {question}")
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to('cuda')
        input_ids = self.tokenizer.encode(text, return_tensors='pt')
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device="cuda")
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            attention_mask=attention_mask,
            max_new_tokens=5,
            do_sample=True,
            top_p=0.9,
            temperature=0.5,
            repetition_penalty=1.1,
            eos_token_id=self.tokenizer.encode('')[0],
            pad_token_id=self.tokenizer.eos_token_id,
        )

        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                         zip(model_inputs.input_ids, generated_ids)]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        if "True" in response:
            response = "True"
        elif "False" in response:
            response = "False"
        else:
            response = "无法确定"
        return response

    def judge(self, dataset) -> List[str]:
        print("[LoraJudger] Judging dataset")
        questions = dataset.question
        all_judgements = [self.infer(question) for question in questions]
        print(f"[LoraJudger] Judgements: {all_judgements}")
        return all_judgements