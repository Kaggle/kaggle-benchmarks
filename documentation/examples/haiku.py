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
# title: Haiku
# ---
# %%

from kaggle_benchmarks import assertions, chats, llm, task, tools

# %%

response = llm.prompt("Write a haiku about Kaggle.")
print(response)

# %%

with chats.new() as chat:
    llm.prompt(
        f"Check whether this haiku has proper format:\n{response}",
        schema=bool,
    )

chat


# %%


@task(name="Haiku verifier")
def haiku(llm) -> bool:
    repl = tools.python.IPythonREPL()
    haiku = llm.prompt("Write a haiku about Kaggle.")

    llm.prompt(
        "Write a python function `is_haiku` that checks whether a haiku has proper format."
    )
    assertions.assert_equal(repl.respond().status, "ok")
    out = repl.invoke(f"assert is_haiku('''{haiku.strip()}''')")

    if out.status == "error":
        llm.prompt(
            "I have tried you function on the haiku you've written and got an error. "
            "Can you fix it? (Do not rename it)"
        )

        assertions.assert_equal(repl.respond().status, "ok")
        out = repl.invoke(f"assert is_haiku('''{haiku.strip()}''')")

    return out.status == "ok"


haiku.run(llm)

# %%
