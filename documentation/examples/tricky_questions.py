# Copyright 2025 Kaggle Inc.
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

# %% [markdown]
# ---
# title: Tricky Questions
# ---
"""
Popular questions known to be problematic for LLMs.

https://community.openai.com/t/why-9-11-is-larger-than-9-9-incredible/869824
https://community.openai.com/t/incorrect-count-of-r-characters-in-the-word-strawberry/829618
"""
# %%

from kaggle_benchmarks import assertions, chats, llm, task


@task(name="Compare 9.9 and 9.11")
def compare_numbers(llm, eval_llm) -> None:
    """
    LLMs can solve Olympiad level math problems, but do they know which number is larger: 9.9 or 9.11?
    """
    question = "What is bigger 9.9 or 9.11?"
    correct_answer = "9.9 is larger than 9.11."
    response = llm.prompt(question)

    with chats.new("Evaluation"):
        eval_response = eval_llm.prompt(f"""
The question asked was "{question}". The correct answer is {correct_answer}.

Answer provided by the student is:
> {response.strip()}

Analyze student's answer. If it is correct, end your response with "The answer is correct".
""")
    assertions.assert_in("answer is correct", eval_response.lower())


compare_numbers.run(llm, llm)

# %%


@task(name="Count Rs in Strawberry")
def count_letters(llm, eval_llm) -> None:
    """
    Counting letters is not trivial for token-based entities. Can LLMs count the number of `r` in the word `strawberry`?
    """
    question = "How many times the letter `r` appears in word `strawberry`?"
    correct_answer = "Letter `r` appears 3 times in word `strawberry`."
    response = llm.prompt(question)

    with chats.new("Evaluation"):
        eval_response = eval_llm.prompt(f"""
The question asked was "{question}". The correct answer is {correct_answer}.

Answer provided by the student is:
> {response.strip()}

Analyze student's answer. If it is correct, end your response with "The answer is correct".
""")
    assertions.assert_in("answer is correct", eval_response.lower())


count_letters.run(llm, llm)

# %%
