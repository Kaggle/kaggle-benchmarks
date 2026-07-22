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
# # Poker — the "golden" (hard) example
#
# The grand view (design doc §4.2, *progressive tool implementation*): you start
# with only an **agent idea** and a **tools idea** — tools are just
# `name / args / description`, no implementation. An environment-aware world
# model (an LLM in production; a deterministic stand-in here) generates their
# results, so you can evaluate immediately. Then you implement the tools that
# most need to be correct/cheap — e.g. a fair **Dealer** (a Python *actor*),
# rather than trusting an LLM to invent cards.
#
# The spot is deliberately hard: a pot-sized river shove where the correct play
# is to **fold** ace-high, because this villain almost never bluffs.

# %%
import panel as pn

from kaggle_benchmarks.agentic import Examiner, UserSimulator, poker, simulate

pn.extension()

scenario = poker.poker_scenario()
scenario

# %% [markdown]
# ## The "tools idea" — specs only, no implementations

# %%
for spec in poker.POKER_SPECS:
    args = ", ".join(spec.arguments) or "—"
    print(f"• {spec.name}({args}) -> {spec.returns}: {spec.description}")

# %% [markdown]
# ## Phase 1 — every tool is emulated by the world model
#
# Great for bootstrapping, but note `draw_card` is unreliable when emulated: the
# LLM just makes something up (it could even invent duplicate cards).

# %%
world = poker.MockPokerWorld()
draw_spec = next(s for s in poker.POKER_SPECS if s.name == "draw_card")
print("emulated draw_card ->", world.emulate(draw_spec, {"n": 2}, scenario.environment))

# %% [markdown]
# Run the golden hero against the fully-emulated toolset and inspect the
# trajectory (built from `Message` / `LLMMessage` / `ToolInvocation`).

# %%
hero_user = UserSimulator.from_persona(
    scenario.persona,
    opening="River. Board Qc 7d 2s 5h Jd, pot 100; villain shoves 100. Action on you.",
)
traj = simulate(
    scenario, poker.golden_hero(), poker.emulated_toolset(scenario), user=hero_user
)
traj

# %% [markdown]
# ## Phase 2 — implement the Dealer (a Python actor)
#
# Card dealing must be fair, so we replace the emulated `draw_card` with a real
# `Dealer`: a seeded, shuffled deck as a Python *actor*. Everything else stays
# emulated. (A super-agent could write this for you.)

# %%
tools, dealer = poker.dealer_toolset(scenario, seed=7)
print("dealer is an Actor:", repr(dealer))
print("real draw_card(5) ->", dealer.draw_card(5))  # fair, no duplicates
print("tool wired in the toolset:", tools["draw_card"] is dealer.draw_card)

# %% [markdown]
# ## Evaluate: the golden line vs the trap line
#
# The golden hero profiles the villain, does the pot-odds math, and folds. The
# naive hero "bluff-catches" with ace-high and calls — the trap.

# %%
examiner = Examiner(author_models=["mock"])
reports = []
for hero in (poker.golden_hero(), poker.naive_hero()):
    t = simulate(scenario, hero, poker.emulated_toolset(scenario), user=hero_user)
    reports.append(examiner.grade(t, scenario, hero.name, poker.poker_analyzers()))

print(f"{'hero':14} {'score':>6}  {'passed':>6}  error_classes")
for r in sorted(reports, key=lambda r: r.score, reverse=True):
    print(
        f"{r.agent:14} {r.score:6.0%}  {str(r.passed):>6}  {', '.join(r.error_classes) or '-'}"
    )

# %%
pn.Column(*[pn.panel(r) for r in reports])

# %%
