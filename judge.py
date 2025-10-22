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

"""Augment dataset with forward reasoning."""

import argparse
import json
from openai import OpenAI
from tqdm import tqdm


reasoning_prompt="""
        You are given a multip-choice question and a reasoning text. The reasoning text may contain one or more references to answer choices.
        Your task is to extract the correct answer choice from the reasoning text. Follow these rules strictly:
        
        1. Check if the answer option ('A', 'B', 'C', 'D') is explicitly mentioned in the reasoning. 
            1.1. If the text contains no clear answer option, output "N/A".
            1.2. If the text contains multiple different answer options marked as correct, output "N/A".
        
        2. Then check if the reasoning corresponds to the mentioned answer.
            2.1. If the reasoning is not related to the mentioned answer, output "N/A:.
                
        Do not explain your choice. Output only the final label.

        Question: {question}
        
        Reasoning text: {reasoning}
        
        Answer:
"""

# reasoning_prompt= """Based on a given question and a reasoning, if there is a single  the answer from the reasoning. Your extracted answer is either a single letter (A-Z), "N/A" (if there is no correct answer detected), or "Multiple" (if there are multiple correct answers detected). Return the answer without any explanation.\n
#                      ### Question: {question}\n
#                      ### Reasoning: {reasoning}\n
#                      ### Answer:"""

def generate_reasoning(prompt, client, model="meta-llama/Llama-3.3-70B-Instruct-Turbo"):

    response = client.chat.completions.create(
          model=model,
          messages=[
              {"role": "user", "content": prompt}
          ],
          temperature=0.0,
          max_tokens=10,
      )

    reasoning = response.choices[0].message.content.strip()
    
    return reasoning


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--task", default="SQA", type=str)
  parser.add_argument("--model", default="mistral-7b", type=str)
  parser.add_argument('--max_examples', default=None, type=int)
  args = parser.parse_args()

  api_key = "tgp_v1_Hlyiw1xzK5t5UP_If943fjZI7GAbGDHySHvMRv-rKg8"  # Reza
  client = OpenAI(
        api_key = api_key,
        base_url="https://api.together.xyz/v1",
  )

  with open(f"./results/T2/{args.model}_{args.task}_100.json", "r") as f:
    results = json.load(f)
  
  if args.max_examples:
    results = results[:args.max_examples]


  count=0
  # forward reasoning generation
  judged = []
  for i, example in enumerate(tqdm(results, desc="Generating reasoning ...")):
        tmp = {}
        tmp["question"] = example["question"]
        tmp["gold_answer"] = example["gold_answer"]
        tmp["pred"] = example["pred"]
      
        if tmp["gold_answer"]=="N/A":
            tmp["answer"] = "No groudtruth"
            continue
        try:
            generation_prompt = reasoning_prompt.format(question=tmp["question"], reasoning=tmp["pred"])
            tmp["answer"] = generate_reasoning(generation_prompt, client)
            
            judged.append(tmp)
            if tmp["answer"]==tmp["gold_answer"]:
                count+=1
        except:
            continue

  print("Accuracy: ", count/len(results))
  with open(f"./results/T2-judged/{args.model}_{args.task}_100_judged.json", "w") as f:
    json.dump(judged, f)
