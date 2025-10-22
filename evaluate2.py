# coding=utf-8
# Copyright 2025 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluate the model on test set."""

import argparse
import json
import os
from huggingface_hub import login

import transformers
from utils import evaluate
from utils import get_alphabet_choice
from utils import get_yes_no
from math_utils import parse_math_boxed
from math_utils import parse_number
from vllm import LLM
from vllm import SamplingParams
from vllm.lora.request import LoRARequest

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", default=0, type=int)
    parser.add_argument("--task", default="SQA", type=str)
    parser.add_argument("--model", default="mistral-7b", type=str)
    parser.add_argument("--model_dir", default="", type=str)
    parser.add_argument("--lora_rank", default=32, type=int)
    args = parser.parse_args()
    
    if args.model == "mistral-7b":
        base_model = "mistralai/Mistral-7B-Instruct-v0.3"
    elif "Llama-3." in args.model:
        base_model = f"meta-llama/{args.model}-Instruct"
    elif "Llama-3" in args.model:
        base_model = f"meta-llama/Meta-{args.model}-Instruct"
    elif "gemma" in args.model:
        base_model = f"google/{args.model}-it"
    elif args.model == 'olmo-2-1b':
        base_model = 'allenai/OLMo-2-0425-1B-Instruct'
    elif args.model == 'olmo-2-7b':
        base_model = 'allenai/OLMo-2-1124-7B-Instruct'
    elif args.model == 'olmo-2-13b':
        base_model = 'allenai/OLMo-2-1124-13B-Instruct'
    elif args.model == 'qwen-2.5-7b':
        base_model = 'Qwen/Qwen2.5-7B'
    else:
        raise ValueError(f"Unsupported model: {args.model}")
        
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        padding_side="right"
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.add_bos_token = False
    tokenizer.add_eos_token = False

    with open(f"./data/T2/{args.model}_{args.task}_test.json", "r") as f:
    # with open(f"./results/{args.model}_{args.task}_100.json", "r") as f:
        # # with open(f"./data/test_data_reordered/ARC_test_reordered.json", "r") as f:
        # # with open(f"./data/test_data_n_false/ARC_test_2_false.json", "r") as f:
        # with open(f"./data/mmlu/virology.json", "r") as f:
        test_samples = json.load(f)

    reasoning_prompt = """Briefly explain what is the answer to the following question:\n\nQuestion:{question}"""
  
    # Use unified chat template approach
    def create_prompt(sample):
        messages = [
            {"role": "user", "content": reasoning_prompt.format(question=sample['question'])}
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    prompts = [create_prompt(s) for s in test_samples]


    sampling_params = SamplingParams(n=1,
                                    temperature=0,
                                    max_tokens=1024,
                                    stop_token_ids=[tokenizer.eos_token_id])
    
    llm = LLM(model=base_model,
            enable_lora=False,
            download_dir=args.model_dir,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.4)
            # max_model_len=8192)
    
    outputs = llm.generate(prompts,
                            sampling_params,
                            lora_request=None)

    # is_math = False
    # if args.task in ["SQA", "BoolQ"]:
    #     answer_extraction = get_yes_no
    # elif args.task in ["ANLI", "ARC", "Date", "CSQA", "OBQA", "ESNLI", "MCGSM8K", "MMLU", "virology", "misc", "stem"]:
    #     answer_extraction = get_alphabet_choice
    # elif args.task in ["GSM8K", "GSM8K-Rev"]:
    #     answer_extraction = parse_number
    #     is_math = True
    # elif args.task in ["TabMWP", "MATH"]:
    #     answer_extraction = parse_math_boxed
    #     is_math = True
    # else:
    #     raise ValueError(f"Unsupported task: {args.task}")

    for e, output in enumerate(outputs):
        try:
            test_samples[e]["pred"] = json.loads(output.outputs[0].text)
        except:
            test_samples[e]["pred"] = output.outputs[0].text
    # acc = evaluate(test_samples, is_math=is_math, pred_key="pred")
    # print(f"accuracy on {args.task}: {acc}")

    os.makedirs("./results/T2/", exist_ok=True)
    save_path = f"./results/T2/{args.model}_{args.task}_{args.n}.json"
    with open(save_path, "w") as f:
        json.dump(test_samples, f)
