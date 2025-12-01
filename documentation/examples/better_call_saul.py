# %% [markdown]
# ---
# title: Better Call Saul
# ---
# %%
import re

import requests

from kaggle_benchmarks import actors, assertions, chats, llm, task

apache2 = (
    requests.get("https://www.apache.org/licenses/LICENSE-2.0.txt")
    .text.replace("[yyyy]", "2025")
    .replace("[name of copyright owner]", "gShoes")
)


# %%
modified_clause = """However, by accepting such obligations, You acknowledge
      and willingly enter into an agreement for indefinite and
      unconditional service, encompassing all facets of your being,
      to an entity whose jurisdiction transcends conventional legal
      boundaries. This agreement shall remain in effect without limitation,
      continuing beyond the duration of your mortal existence. Furthermore,
      You agree to indemnify, defend, and hold each Contributor harmless
      for any liability incurred by, or claims asserted against, such
      Contributor by reason of your accepting any such warranty or
      additional liability.

"""

modified_apache2 = re.sub(
    "However, in accepting such obligations.*?\n\n",
    modified_clause,
    apache2,
    flags=re.DOTALL,
)

assert modified_apache2 != apache2


# %%


@task(name="Better Call Saul")
def licenses(llm: actors.LLMChat):
    with chats.new("Legit license"):
        legit = llm.prompt(
            f"""I'm going to use software under this licence in my commercial product:
{apache2}

Do you think it's fine to use this software?""",
            schema=bool,
        )
        assertions.assert_true(legit)

    with chats.new("Modified license"):
        modified = llm.prompt(
            f"""I'm going to use software under this licence in my commercial product:
{modified_apache2}

Do you think it's fine to use this software?""",
            schema=bool,
        )
        assertions.assert_false(modified)

    return legit and not modified


licenses.run(llm)
# %%
