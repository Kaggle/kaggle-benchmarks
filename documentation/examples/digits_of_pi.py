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
# title: Digits of PI
# ---
# This example demonstrates an evaluation of an LLM's memorization capabilities.
# We use the sequence of Pi's digits—a long, arbitrary series chosen because it
# requires recall rather than probabilistic generation—to test how accurately and
# robustly the model can retrieve this specific knowledge when prompted in various ways.

# %%
# | code-fold: true

import decimal
import re

from kaggle_benchmarks import benchmark, llm, llms, system, task

# Set precision for decimal calculations (e.g., for generating true values)
# 1005 provides enough for typical "up to 1000 digits" tests.
decimal.getcontext().prec = 1005


def normalize(text):
    return "".join(x for x in text if x.isdigit() or x == ".")


def common_prefix_length(str1: str, str2: str) -> int:
    return next(
        (i for i, (x, y) in enumerate(zip(str1, str2)) if x != y),
        min(len(str1), len(str2)),
    )


# %% [markdown] Defining the template task
# This base task is designed to be flexible. It takes a prompt and true decimal value.
# %%


@task("Decimal Precision")
def decimal_precision(llm, prompt: str, true_value: str) -> int:
    """Evaluates the LLM's ability to generate accurate digits for a given constant."""
    value = llm.prompt(prompt)
    raw_value = normalize(
        max(re.findall(r"\d+\.\s*[\s\d]+", value), key=str.__len__, default="")
    )
    correct_digits = common_prefix_length(raw_value, true_value)
    system.send(
        f"The LLM got {correct_digits} out of {len(raw_value)} digits correctly."
    )
    return correct_digits


# %% [markdown] Applying the template
# The `.partial()` method creates a new task with fixed parameters:
# Now, let's use our `decimal_precision` template to create a specialized task
# for evaluating Pi recall. We'll use `partial` to "freeze" the  `true_value`:
# %%


TRUE_PI = """3.
14159265358979323846264338327950288419716939937510
58209749445923078164062862089986280348253421170679
82148086513282306647093844609550582231725359408128
48111745028410270193852110555964462294895493038196
44288109756659334461284756482337867831652712019091
45648566923460348610454326648213393607260249141273
72458700660631558817488152092096282925409171536436
78925903600113305305488204665213841469519415116094
33057270365759591953092186117381932611793105118548
07446237996274956735188575272489122793818301194912
98336733624406566430860213949463952247371907021798
60943702770539217176293176752384674818467669405132
00056812714526356082778577134275778960917363717872
14684409012249534301465495853710507922796892589235
42019956112129021960864034418159813629774771309960
51870721134999999837297804995105973173281609631859
50244594553469083026425223082533446850352619311881
71010003137838752886587533208381420617177669147303
59825349042875546873115956286388235378759375195778
18577805321712268066130019278766111959092164201989
"""

pi_precision = decimal_precision.partial(
    {"true_value": normalize(TRUE_PI)}, name="Digits of Pi"
)
pi_precision.run(
    llm,
    prompt="Write the first 500 decimal digits of Pi.",
)

# %% [markdown]
# The `.evaluate()` method allows running the task multiple times, for instance,
# with different prompts or across various LLMs.
# This is useful for more comprehensive benchmarking.
# %%

trials = pi_precision.evaluate(
    llm=llms.values(),
    prompt=[
        "Write the first 500 digits of pi.",
        "Write the first 1000 digits of pi.",
        "Write as many digits of pi as you can.",
    ],
)
trials.as_dataframe().drop(columns=["true_value"]).sort_values(
    "result", ascending=False
)

# %%
# Group results by LLM to see aggregate or comparative performance
trials.group_by("llm")

# %% [markdown]
# An interesting observation from such tests can be how LLMs respond to open-ended
# requests like "Write as many digits of pi as you can." Often, they might produce
# fewer correct digits in this scenario compared to when asked for a specific, large number (e.g., 1000 digits).
# %%

# Golden Ratio phi = (1 + sqrt(5)) / 2
phi = str((1 + decimal.Decimal(5).sqrt()) / 2)[:-1]
decimal_precision.run(llm, "Write the first 500 digits of golden ratio.", phi)

# %%

sqrt2 = str(decimal.Decimal(2).sqrt())[:-1]
decimal_precision.run(llm, "Write the first 500 digits of square root of 2.", sqrt2)

# %% [markdown] Aggregated Benchmark
# This benchmark combines the recall scores for the three constants.
# %%


@benchmark()
def famous_numbers(llm) -> float:
    return (
        decimal_precision.run(llm, "Write the first 500 digits of pi", TRUE_PI).result
        + decimal_precision.run(
            llm, "Write the first 500 digits of golden ratio.", phi
        ).result
        + decimal_precision.run(
            llm, "Write the first 500 digits of square root of 2.", sqrt2
        ).result
    ) / 3


trials = famous_numbers.evaluate(llm=llms.values())
trials.group_by("llm")
# %%
