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

# %%
# Run the utility function to auto generate code.
task = "What is Kaggle?"
assertion = "Key word 'data science' should be in the answer."
generated_code = kbench.utils.task_autopilot(task, assertion)
print(generated_code)


# %%
# Run the line magic to auto generate code.
# %autopilot


# %%
@kbench.task(
    name="What is Kaggle?", description="Does the LLM know what Kaggle is all about?"
)
def what_is_kaggle(llm) -> bool:
    response: str = llm.prompt("What is Kaggle?")
    answer = response.lower()
    kbench.assertions.assert_in("platform", answer)
    return ("competitions" in answer) and ("community" in answer)


# %%
mp_logger = kbench.utils.log_model_proxy_call()
logged_what_is_kaggle_run = mp_logger(what_is_kaggle.run)
logged_what_is_kaggle_run(kbench.llm)

# %%
# %mplog what_is_kaggle.run(kbench.llm)

# %%
