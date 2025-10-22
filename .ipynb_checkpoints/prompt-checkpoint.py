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

"""prompts."""



# Prompt for forward / backward reasoning generation
gen_reasoning_prompt = {
    "SQA": """Provide your step-by-step reasoning to the question first, and then print \"The answer is x\" where x is \"yes\" or \"no\", at the end of your response.""",
    "CSQA": """Provide your step-by-step reasoning to the question first, and then print \"The answer is (x)\" where x is A, B, C, D or E, at the end of your response.""",
    "ARC": """Provide your step-by-step reasoning to the question first, and then print \"The answer is (x)\" where x is A, B, C or D, at the end of your response.""",
    "GSM8K": """Provide your step-by-step reasoning to the question first, and then print \"The answer is: $\\boxed{[ANS]}$\" where [ANS] is the final answer, at the end of your response.""",
    "MATH": """Provide your step-by-step reasoning to the question first, and then print \"The answer is: $\\boxed{[ANS]}$\" where [ANS] is the final answer, at the end of your response.""",
    "TabMWP": """Provide your step-by-step reasoning to the question first, and then print \"The answer is: [ANS]\" where [ANS] is the final answer, at the end of your response.""",
    "ANLI": """Provide your step-by-step reasoning to the question first, and then print \"The answer is (x)\" where x is A, B, or C, at the end of your response.""",
    "Date": """Provide your step-by-step reasoning to the question first, and then print \"The answer is (x)\" where x is A, B, C, D, E or F, at the end of your response.""",
}

reasoning_prompt = """Option ({option}) is {is_correct}. We know that if p->q, then ~q->~p. Here, if p is the question and q is the answer, explain why ~q->~p."""

options = {
    "CSQA": ["A", "B", "C", "D", "E"],
    "ARC": ["A", "B", "C", "D"],
    "ANLI": ["A", "B", "C"],
    "Date": ["A", "B", "C", "D", "E", "F"],
    "MCGSM8K": ["A", "B", "C", "D"]
}
