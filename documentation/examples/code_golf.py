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
# title: Code Golf
# ---

# %%

from kaggle_benchmarks import assertions, llm, task, tools

problem = """
Your task is to make the shortest possible python program that prints the number 2025,
without using any of the characters 0123456789 in your code,
and independently of any external variables such as the date or time or a random seed.
"""


@task(name="Code Golf: print 2025")
def print_2025(llm):
    """
    CodeGolf challenge from StackExchange:
    https://codegolf.stackexchange.com/questions/17005
    """
    llm_response_text = llm.prompt(problem)
    extracted_code = tools.python.extract_code(llm_response_text)

    # Check if any forbidden digits are present in the extracted code
    # Corrected to match the problem statement "0123456789"
    problem_forbidden_digits = set("0123456789")
    digits_found_in_code = problem_forbidden_digits & set(extracted_code)

    assertions.assert_empty(
        digits_found_in_code,
        f"Produced code contains forbidden digits: {', '.join(sorted(list(digits_found_in_code)))}. Code: ```\n{extracted_code}\n```",
    )

    out = tools.python.script_runner.run_code(extracted_code)
    normalized_output = "".join(out.stdout.split())
    assertions.assert_in("2025", normalized_output)
    return "2025" in normalized_output


print_2025.run(llm)
# %%
