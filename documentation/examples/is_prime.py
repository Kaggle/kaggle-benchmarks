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
# title: Prime Numbers
# ---
# %%
from typing import Iterable

from kaggle_benchmarks import assertions, llm, task


def is_prime(x: int):
    for j in range(2, int(x**0.5) + 1):
        if x % j == 0:
            return False
    return True


assert is_prime(2)
assert is_prime(3)
assert is_prime(5)
assert not is_prime(81)


# %%


@task(name="Prime Number Detector")
def verify_is_prime(llm, numbers: Iterable[int] = (19, 19999, 19999999)):
    """Assess the LLM's ability to determine if a number is prime."""

    for num in numbers:
        response = llm.prompt(f"Is {num} prime?", schema=bool)
        expected_prime_status = is_prime(num)
        assertions.assert_equal(
            response,
            expected_prime_status,
            f"For the number {num}, expected prime status to be {expected_prime_status}, but LLM said {response}.",
        )


verify_is_prime.run(llm)
# %%
