# Copyright 2026 Kaggle Inc.
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
# # The Examiner: author a Suite (with rotation + de-dup)
#
# Design doc §4.1 (domain → scenarios) and §10 (rotate author models, drop
# near-duplicates, record provenance). Generation is mocked
# (`kaggle_benchmarks.agentic.demo.author_one`); the rotation/dedup/storage is
# real. `Scenario` and `Suite` are pydantic models.

# %%
import tempfile

import panel as pn

from kaggle_benchmarks.agentic import Examiner, Suite, diversity_report
from kaggle_benchmarks.agentic.demo import author_one

pn.extension()

# %% [markdown]
# ## Author a suite
#
# Five scenarios are generated, rotating three author models. The demo's cities
# cycle so #0 and #3 collide — one gets dropped as a near-duplicate.

# %%
examiner = Examiner(author_models=["mock-pro", "mock-flash", "mock-sonnet"])
suite = examiner.author(
    problem=(
        "Evaluate a travel-planning agent that must weigh weather, price, and "
        "local events when recommending trip dates."
    ),
    author_one=author_one,
    n=5,
    seed=7,
)
suite.metadata

# %% [markdown]
# The `Suite` renders itself, and `diversity_report` summarizes tag coverage and
# author distribution.

# %%
print("version:", suite.version())
print("diversity:", diversity_report(suite))
suite

# %% [markdown]
# ## Freeze it to disk (humans can edit the JSON, add tasks)

# %%
path = tempfile.mktemp(suffix=".suite.json")
suite.save(path)
reloaded = Suite.load(path)
print(
    f"reloaded {len(reloaded)} scenarios; version matches: {reloaded.version() == suite.version()}"
)

# %%
