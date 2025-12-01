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
import kaggle_benchmarks as kbench
from kaggle_benchmarks.content_types import images
from kaggle_benchmarks.kaggle import models


@kbench.task(name="describe_image")
def describe_image(llm):
    image_url = (
        # "https://upload.wikimedia.org/wikipedia/commons/4/47/American_Eskimo_Dog.jpg"
        "https://upload.wikimedia.org/wikipedia/commons/d/d7/Mammaliatheirva00figu_orig_0227.png"
    )
    response = llm.prompt(
        "What is the animal in the picture?", image=images.from_url(image_url)
    )
    kbench.assertions.assert_contains_regex(
        "(?i)horse",
        response,
        expectation=f"LLM should identify the object correctly. Expected: horse, Got: {response}",
    )


# %%
# Model with default API
describe_image.run(kbench.llm)


# %%
# Model with genai API
llm_with_genai_api = models.load_model(
    model_name=kbench.llm.name,
    api="genai",
)

describe_image.run(llm_with_genai_api)

# %%
