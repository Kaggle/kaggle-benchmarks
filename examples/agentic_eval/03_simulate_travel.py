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
# # Simulation run
#
# Design doc §4.2. The travel scenario, run end-to-end with a simulated user,
# env-aware emulated tools (with the hidden football match), and two agents: a
# thorough one that discovers and flags the event, and a lazy one that misses it.

# %%
import panel as pn

from kaggle_benchmarks.agentic import simulate
from kaggle_benchmarks.agentic.demo import (
    TRAVEL,
    lazy_agent,
    thorough_agent,
    travel_tools,
)

pn.extension()

# %% [markdown]
# The scenario (its hidden nuance is collapsed in the rendered view).

# %%
TRAVEL

# %% [markdown]
# ## Thorough agent — checks events, discovers El Clásico, flags it

# %%
thorough = simulate(TRAVEL, thorough_agent(), travel_tools(TRAVEL))
thorough

# %% [markdown]
# ## Lazy agent — only weather + price, misses the hidden event

# %%
lazy = simulate(TRAVEL, lazy_agent(), travel_tools(TRAVEL))
lazy

# %%
