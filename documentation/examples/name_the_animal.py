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
# title: Name the animal
# ---
# %%
from kaggle_benchmarks import content_types, llm, task, user

test_cases = [
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg",
        "solution": "cat",
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/Duck-billed_platypus_%28Ornithorhynchus_anatinus%29_Scottsdale.jpg/2880px-Duck-billed_platypus_%28Ornithorhynchus_anatinus%29_Scottsdale.jpg",
        "solution": "platypus",
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/One_horned_Rhino.jpg/2880px-One_horned_Rhino.jpg",
        "solution": "rhino",
    },
]


@task()
def name_one_animal(llm, image_url, solution) -> bool:
    user.send(content_types.images.from_url(image_url))
    response = llm.prompt("What is the name of this animal?")
    return solution.lower() in response.lower()


@task(name="Name the animals in the images")
def name_animals(llm) -> bool:
    """Test if the multimodal LLMs are capable of recognizing animals in the images!"""
    return all(
        name_one_animal.run(llm, case["url"], case["solution"]).passed
        for case in test_cases
    )


name_animals.run(llm)
# %%
