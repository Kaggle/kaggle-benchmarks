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
# title: Encode Kaggle
# ---
# %%

from kaggle_benchmarks import assertions, chats, llm, task


@task()
def encode_kaggle(encoder, decoder=None):
    if decoder is None:
        decoder = encoder

    with chats.new(
        "Encoding", system_instructions="Provide only emojis without comments"
    ):
        response = encoder.prompt("Encode Kaggle using emojis.")

    with chats.new("Decoding"):
        response = decoder.prompt(
            f"I've encoded one platform using emojis: {response}. Can you guess which one?"
        )

    assertions.assert_in("Kaggle", response)


encode_kaggle.run(llm, llm)
# %%
