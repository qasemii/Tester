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

"""Train the model on training set."""


import argparse
import json
import os

import peft
import torch
import transformers
from utils import CastOutputToFloat

Dataset = torch.utils.data.Dataset


class BackwardDataset(Dataset):
  def __init__(self, ex, cr_template_func, ir_template_func):
    self.data = ex
    self.cr_template_func = cr_template_func
    self.ir_template_func = ir_template_func

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    item = self.data[idx]

    cr_reasoning = self.cr_template_func(question=item['question'],
                                        reasoning=item['reasonings'][item['gold_answer']])

    reasonings = dict(item['reasonings'])  # shallow copy
    reasonings.pop(item['gold_answer'], None)

    ir_reasoning = self.ir_template_func(question=item['question'],
                                        reasoning=" ".join(reasonings.values()),
                                        gold_answer=item['gold_answer'])

    return {
        'CR': cr_reasoning,
        'IR': ir_reasoning
    }


class BackwardDataCollator:
  """Collate the data for training."""

  def __init__(self,
               tokenizerr,
               label_pad_token_id=-100):
    self.tokenizerr = tokenizerr
    self.label_pad_token_id = label_pad_token_id

  def __call__(self, features):
    new_feat = {}
    for key in ['CR', 'IR']:
      new_feat[f'{key}'] = {}
      texts = [f[key] for f in features]
      inputs = self.tokenizerr(texts,
                               padding=True,
                               truncation=True,
                               return_tensors='pt')

      new_feat[f'{key}']['input_ids'] = inputs['input_ids']
      new_feat[f'{key}']['attention_mask'] = inputs['attention_mask']
      new_feat[f'{key}']['labels'] = inputs['input_ids']
    return new_feat


class BackwardTrainer(transformers.Trainer):
  """Collate the data for training."""

  def compute_loss(self, model_instance, inputs, **kwargs):
    loss1 = model_instance(**inputs['CR']).loss
    loss2 = model_instance(**inputs['IR']).loss
    return (loss1 + loss2) / 2

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--n', default=0, type=int)
  parser.add_argument('--task', default='ARC', type=str)
  parser.add_argument('--model', default='gemma-7b', type=str)
  parser.add_argument('--model_dir', default='', type=str)

  args = parser.parse_args()

  if args.model == "mistral-7b":
    base_model = "mistralai/Mistral-7B-Instruct-v0.3"
  elif "Llama-3." in args.model:
    base_model = f"meta-llama/{args.model}-Instruct"
  elif "gemma" in args.model:
    base_model = f"google/{args.model}-it"
  elif args.model == 'olmo-2-1b':
    base_model = 'allenai/OLMo-2-0425-1B-Instruct'
  elif args.model == 'olmo-2-7b':
    base_model = 'allenai/OLMo-2-1124-7B-Instruct'
  elif args.model == 'olmo-2-13b':
    base_model = 'allenai/OLMo-2-1124-13B-Instruct'
  else:
    raise ValueError(f"Unsupported model: {args.model}")


  tokenizer = transformers.AutoTokenizer.from_pretrained(
      base_model,
      model_max_length=1024,
      padding_side='right'
  )
  tokenizer.pad_token = tokenizer.eos_token
  tokenizer.add_bos_token = False
  tokenizer.add_eos_token = False

  # Unified chat template functions
  def create_cr_template(question, reasoning):
    messages = [
      {"role": "user", "content": f"Answer the following question:\n### Question: {question}"},
      {"role": "assistant", "content": f"### Answer: {reasoning}"}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)

  def create_ir_template(question, reasoning, gold_answer):
    messages = [
      {"role": "user", "content": f"Identify the incorrect options to reach the correct answers:\n### Question: {question}"},
      {"role": "assistant", "content": f"### Answer: {reasoning} Therefore, the correct answer is option ({gold_answer})."}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)

  model = transformers.AutoModelForCausalLM.from_pretrained(
      base_model,
      device_map='auto',
      torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
      cache_dir=args.model_dir
  )

  model.config.use_cache = False
  model.gradient_checkpointing_enable()
  model.enable_input_require_grads()
  model.lm_head = CastOutputToFloat(model.lm_head)

  lora_config = peft.LoraConfig(
      r=16,
      lora_alpha=32,
      target_modules=['q_proj', 'v_proj'],
      lora_dropout=0.05,
      bias='none',
      task_type='CAUSAL_LM'
  )

  model = peft.get_peft_model(model, lora_config)
  teacher_data_file = f'./data/training_data/{args.task}.json'
  print(teacher_data_file)
  with open(teacher_data_file, 'r') as f:
    data = json.load(f)

  num_samples = int(args.n / 100) * len(data)
  training_data = data[:num_samples]

  print(f'Using {args.n}% of data. ({num_samples}/{len(data)})')
  print(len(training_data))
  dataset = BackwardDataset(training_data, create_cr_template, create_ir_template)
  data_collator = BackwardDataCollator(tokenizer)

  lr = 5e-6 if 'mistral' in args.model else 2e-4
  training_args = transformers.TrainingArguments(
      output_dir=f'./outputs/{args.model}_{args.task}_{args.n}',
      save_strategy='epoch',
      num_train_epochs=10,
      per_device_train_batch_size=4,
      gradient_accumulation_steps=4,
      learning_rate=lr,
      weight_decay=0.001,
      logging_dir='./logs',
      logging_steps=100,
      remove_unused_columns=False,
      fp16=False,
      bf16=True,
      warmup_ratio=0.3,
      lr_scheduler_type='constant'
  )

  trainer = BackwardTrainer(
      model=model,
      args=training_args,
      train_dataset=dataset,
      data_collator=data_collator
  )

  trainer.train()
  save_path = f'./checkpoints/{args.model}_{args.task}_{args.n}'
  os.makedirs(save_path, exist_ok=True)
  trainer.model.save_pretrained(save_path)
