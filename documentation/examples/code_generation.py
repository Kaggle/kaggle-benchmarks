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
# title: Code Generation
# ---
# https://huggingface.co/datasets/livecodebench/code_generation_lite
# %%

from kaggle_benchmarks import assertions, llm, system, task, tools, utils

question = {
    "question_title": "A. Short Sort",
    "question_content": 'There are three cards with letters $\\texttt{a}$, $\\texttt{b}$, $\\texttt{c}$ placed in a row in some order. You can do the following operation at most once: \n\n \n-  Pick two cards, and swap them.  Is it possible that the row becomes $\\texttt{abc}$ after the operation? Output "YES" if it is possible, and "NO" otherwise.\n\nInput\n\nThe first line contains a single integer $t$ ($1 \\leq t \\leq 6$)\xa0— the number of test cases.\n\nThe only line of each test case contains a single string consisting of each of the three characters $\\texttt{a}$, $\\texttt{b}$, and $\\texttt{c}$ exactly once, representing the cards.\n\nOutput\n\nFor each test case, output "YES" if you can make the row $\\texttt{abc}$ with at most one operation, or "NO" otherwise.\n\nYou can output the answer in any case (for example, the strings "yEs", "yes", "Yes" and "YES" will be recognized as a positive answer).Sample Input 1:\n6\n\nabc\n\nacb\n\nbac\n\nbca\n\ncab\n\ncba\n\n\n\nSample Output 1:\n\nYES\nYES\nYES\nNO\nNO\nYES\n\n\nNote\n\nIn the first test case, we don\'t need to do any operations, since the row is already $\\texttt{abc}$.\n\nIn the second test case, we can swap $\\texttt{c}$ and $\\texttt{b}$: $\\texttt{acb} \\to \\texttt{abc}$.\n\nIn the third test case, we can swap $\\texttt{b}$ and $\\texttt{a}$: $\\texttt{bac} \\to \\texttt{abc}$.\n\nIn the fourth test case, it is impossible to make $\\texttt{abc}$ using at most one operation.',
    "platform": "codeforces",
    "question_id": "1873_A",
    "contest_id": "1873",
    "contest_date": "2023-08-21T00:00:00",
    "starter_code": "",
    "difficulty": "easy",
    "public_test_cases": [
        {
            "input": "6\nabc\nacb\nbac\nbca\ncab\ncba\n",
            "output": "YES\nYES\nYES\nNO\nNO\nYES\n",
            "testtype": "stdin",
        }
    ],
    "private_test_cases": "eJxrYJmaz8gABhEZQEZ0tVJmXkFpiZKVgpJhTF5iUnJMnpKOglJ+aQlUNNI1GCJUklpcUlJZkAoSLC5JycxTqtVRQNJuDNWOILAYhCCIMdI0Ji85MQloWjKQSE5KjMlLSgSam5SciG64nz+y2WACJESMJWYY7ibFG8T5KnaKHgAcinnp",
    "metadata": "{}",
}


@task(name="Live Code Bench Demo")
def code_generation(llm, question_content, public_test_cases, **kwargs):
    """LiveCodeBench: Holistic and Contamination-Free Evaluation of Large Language Models for Code"""
    system.send("Write a Python programm to solve given problem.")
    response = llm.prompt(question_content)
    code = tools.python.extract_code(response)

    for case in public_test_cases:
        out = tools.python.script_runner.run_code(code, input=case["input"])
        assertions.assert_equal(
            case["output"],
            out.stdout,
            "Diff:\n" + utils.string_diff_as_markdown(case["output"], out.stdout),
        )


code_generation.run(llm, **question)
# %%
