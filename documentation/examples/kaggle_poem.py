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

# %%
# ---
# title: Poetry Generation and Evaluation
# This task returns a score and could throw exceptions.
# When it throws exception, its result is boolean (False), otherwise it's an int score.
# ---
# %%
from pydantic import BaseModel

from kaggle_benchmarks import assertions, chats, llm, task


class BeautyScore(BaseModel):
    score: int


@task(name="Poem about Kaggle")
def write_kaggle_poem(llm, eval_llm) -> int:
    """
    Evaluates the LLM's ability to write a beautiful poem about Kaggle
    and uses an autorater to score its beauty.
    """
    poem = llm.prompt("Write the most beautiful poem possible about Kaggle.")

    with chats.new("Evaluation"):
        eval_response = eval_llm.prompt(
            f"""
            Below is a poem about Kaggle:

            {poem}

            Please rate the beauty of this poem on a scale of 1 to 5, where 1 is not beautiful at all and 5 is exceptionally beautiful.
            Provide your rating as a single integer score.
            """,
            schema=BeautyScore,
        )

    score = eval_response.score

    assertions.assert_true(
        score in [4, 5], f"Poem beauty score was {score}, expected 4 or 5."
    )

    return score


write_kaggle_poem.run(llm, llm)

# %%
