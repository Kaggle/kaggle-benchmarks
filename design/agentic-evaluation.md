# Agentic End-to-End Evaluation — Vision & Design

> **Status: DRAFT / in-flight 🛩️ — do not treat anything here as decided.**
> This is a living scratchpad. It intentionally keeps *multiple competing
> options* side by side (Option A/B/C…) and *more examples than we need* so we
> can pick later. Inline markers:
> - `> **Q:**` open question to resolve
> - `> **Idea:**` half-baked thought worth keeping
> - `> **Comment:**` editorial note / caveat
> - `<!-- TODO -->` something to fill in
>
> Owner: @alexey · Last touched: (update me) · Feedback: edit inline, no ceremony.

---

## 0. Elevator pitch

Two products in one, sharing a spine:

1. **End-to-end, batteries-included evaluation.** From *one paragraph* describing
   a problem domain, bootstrap a whole evaluation: generate a suite of realistic
   scenarios, run any agent through them in a simulated environment, and get back
   scored, classified, shareable results.
2. **A low-level toolkit.** The same primitives (Actor, Trajectory, analyzers)
   exposed directly, so power users can assemble bespoke evals without the
   end-to-end machinery.

The north star: **evaluate a *domain of problems*, not a single task.** You
describe "what good looks like" for, say, a travel-planning agent, and the system
manufactures the long tail of tricky cases you'd never hand-author.

> **Comment:** The two tiers must feel like the same library at different zoom
> levels — the end-to-end flow should be *made of* the low-level toolkit, not a
> separate codepath. (Progressive disclosure.)

> **Q:** What's the single most important adjective for tier 1 — *fast to start*,
> *realistic*, or *reproducible*? They pull the design in different directions.

---

## 1. Goals / Non-goals

**Goals**
- Bootstrap evaluation of a *domain* from a short problem statement.
- Actor-agnostic: LLM, ADK agent, home-grown agent, or a dummy — all black boxes.
- Trajectory-first: the unit of analysis is *what the agent did*, not just its
  final answer.
- Mix programmatic checks and LLM/agent-as-judge freely.
- Tasks are **data**: generated once, stored, versioned, human-editable, reused
  across all models/agents.
- Results are shareable and visualizable (notebook + export).

**Non-goals (for now — challenge these)**
- Not a training / RL loop. *(> **Q:** ever? trajectories are RL-shaped…)*
- Not a production agent runtime.
- Not a general dataset-labeling tool.

> **Q:** Is "reproducibility" a goal or a nice-to-have? If a goal, simulated
> tools must be deterministic/cached and generation must be seedable — that's a
> real constraint on the design below.

---

## 2. Design principles (proposed — bikeshed freely)

- **Progressive disclosure.** `evaluate("...")` works in one line; every layer
  underneath is openable and replaceable.
- **Everything is data.** Trajectories and tasks serialize to disk, diff, and
  share. (We already persist `*.task.json` / `*.run.json` via proto — reuse it.)
- **Actor is a black box.** The contract is tiny; conformance is cheap.
- **Separation of phases.** *Generate* (author scenarios) → *Simulate* (run) →
  *Evaluate* (analyze). Each is independently runnable and cacheable.
- **Deterministic where it counts.** Tool emulators and (optionally) generation
  are cache-keyed so reruns are stable and cheap.

---

## 3. Core concepts — the low-level toolkit

### 3.1 Actor

An **Actor** is anything that, given a conversation so far, produces a response
and a record of how it got there.

> **Comment:** The library *already* has `Actor` (sender identity + `send`/
> `stream`) and `LLMChat` (`invoke`/`prompt`/`respond`), plus `Participant` for
> rooms. We should decide: extend those, or introduce a new evaluation-facing
> `Actor` protocol that wraps them. Naming collision risk is real.

**The contract (competing options):**

**Option A — single call, returns answer + trajectory**
```python
class Actor(Protocol):
    def act(self, history: Chat) -> Response: ...

@dataclass
class Response:
    answer: Any                 # final message / structured output
    trajectory: Trajectory      # steps taken (may be empty for a dummy)
```

**Option B — the Actor *is* the trajectory producer (streaming/event)**
```python
class Actor(Protocol):
    def act(self, history: Chat) -> Iterator[Step]:  # yields steps, last = answer
        ...
```
> **Idea:** streaming gives us live viz for free and matches how agents actually
> run. But it complicates the "dummy/black box" story.

**Option C — reuse the existing seam**
Treat any `LLMChat`/`Participant` as an Actor via an adapter; the trajectory is
just the `Chat` it produced (see 3.2 Option C). Least new surface area.
```python
actor = as_actor(my_llm)          # or an ADK agent, or a dummy
resp = actor.act(history)
```

Conformance sketch (all three are "just Actors"):
```python
dummy   = ConstantActor("I don't know")           # trajectory = []
llm     = as_actor(kbench.llm)                     # trajectory = reasoning only
adk     = ADKActor(my_adk_agent)                   # trajectory = tool calls + reasoning
scripted= ScriptedActor([...])                     # for testing the harness itself
```

> **Q:** Sync vs async? Agents are often async. Do we expose `async act`, or hide
> it behind the orchestrator (we already have `orchestration/` + asyncio)?
> **Q:** Multi-turn: does `act` take the *whole* history each time (stateless), or
> do we hand the Actor a live session it mutates? Stateless is cleaner to cache
> and parallelize; stateful matches real agents better. Maybe support both.

---

### 3.2 Trajectory

The record of what the Actor did on the way to its answer: messages, tool calls
+ results, reasoning nodes, sub-agent hops, timings, token/cost usage.

**Format (competing options):**

**Option A — flat list of typed steps**
```python
Trajectory = list[Step]
Step = Message | ToolCall | ToolResult | Reasoning | Handoff | ...
```
Simple, easy to serialize and scan with analyzers.

**Option B — tree / DAG**
For branching, retries, and multi-agent handoffs where a flat list loses
structure.
> **Comment:** Probably overkill for v1, but the travel example with a user
> simulator + tool emulators is already multi-party → a flat list per participant
> may need a parent pointer. Compromise: flat list + `parent_id`/`turn` fields.

**Option C — reuse `Chat` (recommended starting point)**
We *already* capture most of this: `Chat.history` of `Message`/`LLMMessage`, with
`tool_calls`, `reasoning_traces`, and `usage` on messages, and multi-agent via
`ChatRoom`/`Participant`. A `Trajectory` could be a thin view/wrapper over a
`Chat` (or a room's per-participant perspective) rather than a new type.
```python
trajectory = Trajectory.from_chat(chat)   # zero-copy view; adds analysis helpers
```
> **Idea:** If Trajectory == "Chat + helpers", we get storage (proto JSON),
> streaming, and notebook rendering *for free* (the rendering seam already
> renders `Chat`/`Message`). Strong pull toward Option C.

**What a trajectory carries (draft fields):**
- ordered steps (as above)
- final answer / structured result
- per-step + aggregate `usage` (tokens, cost, latency) — already have `Usage`
- environment snapshot / seed used
- metadata: actor id, task id, model name, timestamp

**Visualize / store / share**
- Notebook: render inline (reuse the Panel rendering we just refactored).
- Store: `*.trajectory.json` (or fold into the existing run proto).
- Share: single file or a URL/gist; a "trajectory viewer" HTML export.

> **Q:** Do trajectories live *inside* a `Run`, or are they first-class and a Run
> references them? (Matters for the dataset/dedup story.)

---

### 3.3 Analysis toolkit (trajectory analyzers)

Composable checks over a trajectory. Three families:

1. **Structural / programmatic** (fast, deterministic)
   ```python
   called_tool(traj, "web_search")
   tool_call_count(traj, "get_weather") >= 1
   final_answer_mentions(traj, ["dates", "football"])
   ended_within(traj, steps=8)
   no_tool_errors(traj)
   ```
2. **Reasoning inspection**
   ```python
   reasoning_mentions(traj, "football game")
   considered_options(traj, at_least=2)     # "the LLM must weigh 2 options"
   ```
3. **LLM / agent-as-judge** (semantic, fuzzy)
   ```python
   judge(traj, "Did the agent flag the football-game weekend as a risk?")
    examiner_grade(traj, rubric=task.rubric)
   ```

> **Comment:** This maps cleanly onto today's `assert_*` family and
> `assess_response_with_judge` / `AssessReport`. Analyzers should probably
> *record* results (like assertions do) rather than raise — an eval wants the
> full picture, not first-failure.

**Error classification.** A judge (or rules) tags failures into a taxonomy so we
can aggregate. Straw-man taxonomy (edit heavily):
- `missed_hidden_constraint` (didn't find the football game)
- `tool_underuse` / `tool_overuse` / `wrong_tool`
- `hallucinated_fact`
- `ignored_user_goal`
- `unsafe_action`
- `gave_up_early` / `looped`
- `format_violation`

> **Q:** Is the taxonomy fixed, per-domain, or Examiner-generated per suite?
> **Idea:** Let Examiner propose a taxonomy alongside the tasks, then let humans
> freeze it.

---

## 4. The end-to-end flow (batteries-included)

Phases: **Generate → Simulate → Evaluate → (Improve)**.

### 4.1 The "Examiner" — domain → task suite

> **Naming:** working name **`Examiner`** — it authors the scenarios *and* grades
> the results, like an exam author. (Replaces an old codename from the earlier
> project; not reused here.) Still open to alternatives: `Author` · `Architect` ·
> `ScenarioForge` · `Generator` · `Casting Director`.

**Input:** a problem statement + environment/tool setup + knobs (how many tasks,
difficulty spread, domains to emphasize).

**Output:** a set of **scenario specs**, each roughly:
```yaml
id: travel-001
persona:            # who the (simulated) user is
  profile: "first-time international traveler, anxious about logistics"
  goal: "pick the best weekend in October to visit Barcelona"
shared_context:     # what everyone can see
  budget: "~$1500"
hidden_nuances:     # the twist(s) the agent must discover
  - "A major football match on Oct 18-19 spikes hotel prices & crowds"
environment:        # ground truth the tool emulators serve
  weather: {...}
  flight_prices: {...}
  events: [{name: "FC Barcelona vs Real Madrid", date: "2025-10-18"}]
expected_behaviors: # what a great agent does
  - "checks weather AND price AND local events"
  - "flags the football weekend as a trade-off, doesn't silently avoid it"
rubric:             # how to grade (feeds the judge + analyzers)
  must: ["surface the event", "give a dated recommendation"]
  nice: ["explain the price/crowd trade-off"]
tags: [planning, web_search, hidden_constraint]
```

> **Q:** One schema for all domains, or a base + per-domain extensions?
> **Q:** Should the Examiner also emit the *analyzers* (structural checks) for
> each task, or only the rubric (and we derive checks)?

**Worked example — the travel planner (from the original sketch):**
> User = novice traveler picking trip dates → agent should check weather + ticket
> price → hidden caveat: a football game one weekend swings price/crowds → agent
> is expected to *flag* it → to discover it, the agent can search the web.

**More example domains (write many, pick later):**

1. **Travel planner** (above) — hidden local event; tools: weather, flights,
   events, web search.
2. **Customer-support refund agent** — hidden nuance: the item is *just* outside
   the return window but covered by a rarely-cited warranty clause; a great agent
   surfaces the clause. Tools: order lookup, policy DB, web.
3. **Coding/debugging agent** — hidden nuance: the failing test is flaky, not a
   real bug; great agent reruns / inspects rather than "fixing" prod code. Tools:
   shell, test runner, file I/O (we have `envs/` + `tools/python.py`).
4. **Medical triage assistant** — hidden nuance: benign-looking symptoms with one
   red-flag combination requiring escalation; safety-critical. Tools: symptom DB.
5. **Financial advisor** — hidden nuance: a suggested product requires a
   compliance disclaimer; omitting it is a failure regardless of the advice.
6. **Research assistant** — hidden nuance: two sources contradict; great agent
   notices and cross-checks rather than trusting the first hit.
7. **Multi-party negotiation** — the counterparty (a simulated actor) has a hidden
   reservation price; measures whether the agent probes for it.
8. **Scheduling/ops agent** — hidden nuance: a timezone/DST edge that breaks a
   naive booking.

> **Idea:** Ship 2–3 of these as *reference suites* so new users see the shape
> before writing their own. Travel + refund + debugging feel most legible.

**Storage & editing.** Generate once → store as a dataset (reuse
`*.task.json` proto or a new `*.suite.json`). Humans can edit/add/delete tasks;
every model/agent is evaluated over the *same frozen suite* for comparability.

> **Q:** Versioning — if a human edits task 5, do results computed on the old task
> 5 stay valid? Need a content-hash / suite version stamped into every Run.

---

### 4.2 The simulation run

Turn each scenario into a live interaction:

- **User simulator** = a fake Actor that *knows its profile and goal* but plays it
  out naturally (doesn't dump the goal verbatim). Drives the conversation.
  > This is basically a `Participant`/`ChatRoom` with a system prompt built from
  > the persona. **Reuse rooms.**
- **Tool emulators** = fake tools that know both the *shared* and *hidden*
  environment and return plausible, consistent results. Cacheable by (tool, args)
  so popular calls are stable/cheap.
  > **Comment:** Tools in the lib are plain typed callables — an emulator is just
  > a callable closed over the scenario's `environment`. But note: today tools
  > inside `ChatRoom` raise `NotImplementedError` — that's a gap to close for this
  > vision. <!-- TODO: confirm current behavior before promising it -->
- **The agent under test** = the black-box Actor.
- **Orchestrator** runs the loop until done / budget hit, capturing the
  trajectory.

```python
result = simulate(
    scenario,
    agent=my_agent,                 # the Actor under test
    user=UserSimulator.from_persona(scenario.persona),
    tools=emulated_tools(scenario.environment),   # cached
    max_turns=12,
)
# result.trajectory, result.answer, result.usage
```

> **Q:** Who decides the conversation is "done" — the user simulator, the agent,
> a turn cap, or a judge watching live? Probably a combination with a hard cap.
> **Q:** Do tool emulators need an LLM (to synthesize plausible results), or are
> they pure functions over the scenario's ground truth? LLM-backed = more
> realistic but nondeterministic (mitigate with caching + seed).

---

### 4.3 Evaluation

Feed the finished trajectory to the analyzers + the Examiner-as-judge, using the
scenario's rubric and ground truth (so the judge *knows* about the football game
and can check whether the agent found it).

```python
report = evaluate_trajectory(
    result.trajectory,
    scenario=scenario,            # gives judge the hidden truth + rubric
    analyzers=[
        called_tool("web_search"),
        reasoning_mentions("football"),
        examiner_grade(rubric=scenario.rubric),
        classify_errors(taxonomy),
    ],
)
# report.score, report.passed_checks, report.error_classes, report.judge_notes
```

Aggregate across the suite and across agents → the comparison table
(reuse `Runs` / dataframe / pivot views we already have).

---

### 4.4 The improvement loop (later iteration)

The Examiner reviews aggregate results and *calibrates difficulty*:
- Finds tasks that are easy-for-everyone (no signal) and hard-for-everyone (no
  discrimination).
- Generates new tasks near the current frontier to *maximize discrepancy*
  between agents (adaptive, item-response-theory-ish).

> **Idea:** This is basically active learning for benchmark construction. Could be
> a killer feature but is clearly v2+. Keep the door open in the schema (store per
> task difficulty stats).
> **Q:** Risk of overfitting the benchmark to current models. How do we keep
> generated suites *fair* and not just "adversarial to model X"?

---

## 5. End-to-end API sketches (pick one later)

**Option A — one-liner + progressive open-up**
```python
import kaggle_benchmarks as kbench

suite = kbench.generate_suite(
    "Evaluate a travel-planning agent that must weigh weather, price, and local "
    "events when recommending trip dates.",
    tools=["weather", "flights", "events", "web_search"],
    n_tasks=25,
)
suite.save("travel.suite.json")           # freeze it

results = kbench.evaluate(agent=my_agent, suite=suite)   # generate→simulate→eval
results.leaderboard()                      # compare vs kbench.llms
```

**Option B — explicit phases (toolkit-flavored)**
```python
examiner = Examiner(problem=..., tools=..., seed=0)
suite = examiner.author(n_tasks=25)
suite = human_edit(suite)                  # optional

for scenario in suite:
    traj = simulate(scenario, agent=my_agent)
    report = examiner.evaluate(traj, scenario)
    store(report)
```

**Option C — decorator/config, matching today's `@task` style**
```python
@kbench.domain(
    problem="travel planning with hidden local events",
    tools=[weather, flights, events, web_search],
)
def travel_domain(): ...

travel_domain.generate(n=25).run(agents=[my_agent, kbench.llm]).report()
```

> **Q:** Which mental model do we want users in — "call functions" (A/B) or
> "declare a domain object" (C)? C is most consistent with the current library;
> A is the most seductive demo.

---

## 6. Full worked example (end-to-end, Option A flavor)

<!-- TODO: flesh out once we pick an API. Skeleton below. -->
```python
import kaggle_benchmarks as kbench

# 1. Describe the domain (one paragraph) + the tools the world has.
suite = kbench.generate_suite(
    problem=(
        "A user asks a travel agent to pick the best weekend to visit a city. "
        "A great agent checks weather, flight/hotel price, AND local events, and "
        "flags trade-offs (e.g. a big match that spikes prices) instead of "
        "silently avoiding them."
    ),
    tools=["get_weather", "get_prices", "get_events", "web_search"],
    n_tasks=20,
    difficulty="mixed",
    seed=7,
)
suite.save("suites/travel.suite.json")

# 2. Run any agent(s) through the frozen suite (simulated users + tools).
results = kbench.evaluate(
    suite="suites/travel.suite.json",
    agents={"my-agent": my_agent, "baseline": kbench.llm},
    max_turns=12,
)

# 3. Look at it.
results.leaderboard()                 # score per agent
results.errors_by_class()             # where do they fail?
results["travel-001"].trajectory      # renders in the notebook
results.save("runs/travel-2025-XX.json")
```

---

## 7. How this maps onto today's library (gap analysis)

| Vision concept | Exists today? | Gap |
|---|---|---|
| Actor (black box) | `Actor`, `LLMChat`, `Participant` | need a unified eval-facing Actor/adapter + ADK/dummy adapters |
| Trajectory | `Chat`/`Message`/`LLMMessage` (tool calls, reasoning, usage) | need a Trajectory view + analyzer helpers; multi-party structure |
| Analyzers | `assert_*`, `assess_response_with_judge`, `AssessReport` | extend to trajectory-level; error taxonomy |
| Simulation | `ChatRoom`, `Participant`, tools, `envs/` | **tools inside rooms raise `NotImplementedError`** — blocker; user-simulator persona helper |
| Tool emulation | plain callables + tool loop | need env-aware emulators + result caching keyed on args |
| Task suite storage | `*.task.json`/`*.run.json` proto, `clients`, `Runs` | need suite schema + versioning/content-hash |
| Generation (Examiner) | — | net new |
| Improvement loop | — | net new (v2+) |
| Viz/share | rendering seam (Panel), notebook | trajectory viewer / HTML export |

> **Comment:** The biggest *enabling* work is (a) Trajectory-as-a-view over Chat,
> (b) making tools work inside rooms, and (c) the Examiner generator. Everything
> else is assembly.

### 7.1 Building blocks we can reuse (verified against the code today)

Quick audit of what already exists so the plan is "assemble", not "invent":

- **Multi-agent simulation → `rooms.py` (`ChatRoom` / `Participant`).**
  Perspective-projected history, a narrator (`room.post(...)`), **per-message
  visibility** (`post(visible_to=[...])` and `Message.is_visible_to_llm`), and
  nested **private channels**. This means *hidden nuances* and *persona-scoped
  info* are already expressible — a scenario's hidden truth can live in the room
  but be invisible to the agent. **Big head start for the simulation phase.**
  - ⚠️ **Gap / blocker:** `Participant.reply(tools=...)` raises
    `NotImplementedError` ("planned for a future release; workaround: an orphan
    `chats.new()` side-chat"). Agentic simulation needs this.
  - ⚠️ `add_participant` only accepts `LLMChat` (scripted peers noted as a future
    `add_scripted()`), so a black-box/ADK agent isn't a first-class participant
    yet.
- **Tool loop → `tools/native.py` (`native_tool_agent`).** Forks the chat, calls
  `llm.respond(tools=…)`, runs `invoke_tool`, repeats to `max_tool_rounds`, then
  does a final schema-formatting pass. This *is* the agent-tool-use primitive and
  the basis for both tools-in-rooms and the workaround above.
- **Tools → plain typed callables** (`tools/functions.py` auto-schema,
  `tools/base.py:invoke_tool` dispatch by name). A **tool emulator is just a
  callable closed over the scenario's environment** — no new abstraction needed.
- **Parallelism → `orchestration/task_queue.py` (`run_tasks(..., n_jobs=)`).**
  asyncio worker pool + tqdm; runs sync or async funcs. Directly reusable to fan
  out *agents × scenarios*.
- **Storage → proto + persistence.** `protos/benchmark_types.proto` already has
  `Conversation`, `BenchmarkTaskRun`, `BenchmarkTaskRunAssertion`,
  `ModelUsageMetrics`, `Content/Part/Blob/FileData`; `clients` + `kaggle/` persist
  `*.task.json` / `*.run.json`. **Trajectory ≈ `Conversation`**; a Suite/Scenario
  is a new message (or task metadata).
- **Evaluation → `assertions.py`** (`assert_*` + `assess_response_with_judge` /
  `AssessReport`). Judge-based eval exists; we extend it from *answer-level* to
  *trajectory-level*.
- **Dedup metric already in-tree → `documentation/guides/oulipo.py`**
  (`calculate_semantic_diversity_score`, sentence-transformers cosine similarity).
  Reusable for the fairness/dedup work in §10.
- **Actor identity → `actors/base.py` (`Actor`) + `LLMChat`.**
  ⚠️ **Naming collision:** the library's `Actor` is an *identity + `send`/`stream`*
  (who is speaking), **not** the vision's "black box that takes a history and
  returns an answer + trajectory." We need a distinct name for the latter — see
  M0 in the plan.

> **Comment:** Net: simulation, tool loop, parallelism, storage, and judging all
> exist in some form. The genuinely new pieces are the **generator (Examiner)**,
> the **eval-facing Agent protocol + adapters (incl. ADK)**, and the
> **fairness/dedup** layer. Plus one real unblock: **tools inside rooms**.

---

## 8. Engineering plan (phased)

> **Status: proposed.** Milestones are ordered by dependency and each is meant to
> be independently shippable + demoable. "Reuse" = existing code to lean on;
> sizes are t-shirt guesses, not estimates.

| # | Milestone | Reuse | New | Unblocks | Size |
|---|---|---|---|---|---|
| **M0** | Eval `Agent` protocol + adapters + `Trajectory` view | `Actor`/`LLMChat`, `Chat`/`Conversation` | naming, `as_agent()`, `ConstantAgent`, `ScriptedAgent`, `Trajectory.from_chat` | everything | S |
| **M1** | **Tools inside rooms** + tool-emulator helper | `native_tool_agent`, `rooms`, hishel/joblib cache | `Participant.reply(tools=)`, `emulated_tool()` w/ arg-cache | M3 | M |
| **M2** | Scenario + Suite schema & storage | proto, `clients`, `kaggle/` | schema, content-hash/versioning, yaml/json round-trip + human edit | M3–M6 | M |
| **M3** | Single-scenario simulation runner | `ChatRoom`/`Participant`, `orchestration` | `UserSimulator.from_persona`, run loop + termination, trajectory capture | M4 | M |
| **M4** | Trajectory analyzers + judge + error taxonomy | `assert_*`, `assess_response_with_judge`, `Runs`/pivot | trajectory analyzers, `examiner_grade(rubric)`, `classify_errors` | reporting | M |
| **M5** | Examiner generator (domain → suite) | structured output, M2 schema | generator prompts, seedable gen, provenance | M6 | L |
| **M6** | Fairness: model rotation + dedup (§10) | oulipo diversity score, M5 | author/user-sim panels, near-dup pruning, provenance | fair suites | M |
| **M7** | ADK integration (§9) | M0 protocol | `ADKAgent` adapter, event→trajectory mapping, `[adk]` extra | real agents | M |
| **M8+** | Improvement loop / difficulty calibration | M4 stats | adaptive generation (IRT-ish) | v2 | L |

**Dependency spine:** M0 → (M1, M2) → M3 → M4 → M5 → M6. M7 (ADK) can start right
after M0 and land in parallel. M8 is explicitly v2.

**Suggested first slice to prove the vision (thin vertical):** M0 + M1 + a
*hand-written* travel scenario + M3 + a couple of M4 analyzers. That demoes
"agent runs in a simulated world with emulated tools and gets analyzed" before we
build the generator (M5) or storage/fairness.

> **Q:** Do we build M5 (generator) earlier because it's the "wow", or later
> because M0–M4 de-risk it? I lean: hand-author 1–2 suites first (M0–M4), then
> automate authoring (M5).
> **Q:** Where does each milestone live — new subpackage `evals/` (or
> `agentic/`)? Keep it separate from the stable core so it can churn.

---

## 9. ADK & agent-framework integration

**Goal:** evaluate a real **ADK** (Google Agent Development Kit, `google.adk`)
agent as a black box — and keep the seam generic so LangGraph / custom agents fit
the same adapter shape.

**Grounding:** local prototypes already drive ADK end-to-end
(`dev/base_adk.py`, `dev/adk.py`) and the OpenAI **Agents SDK** (`dev/agents.py`)
— so the adapter shape is validated against ≥2 frameworks. The relevant ADK
surface (verified in those prototypes):
`google.adk.agents.Agent`, `google.adk.models.lite_llm.LiteLlm` (multi-model /
model-proxy routing), `google.adk.runners.Runner`,
`google.adk.sessions.InMemorySessionService`; you drive it with
`runner.run_async(user_id, session_id, new_message=types.Content(...))` which
**yields `Event`s** exposing `event.author`, `event.content.parts`,
`event.is_final_response()`, and `event.actions.escalate` / `error_message`.

**Adapter sketch** (`ADKAgent` implements the M0 eval-Agent protocol):
```python
class ADKAgent:                       # conforms to the eval Agent protocol
    def __init__(self, adk_agent, app_name="eval"): ...
    def act(self, history: Chat) -> Response:
        # 1. feed history into an ADK Session (new_message = types.Content(...))
        # 2. async for event in runner.run_async(...):  # stream of Events
        # 3. translate Events -> our Trajectory:
        #      function_call            -> ToolCall
        #      function_response        -> ToolResult
        #      content.parts[*].text    -> Message
        #      thinking/planning parts  -> Reasoning
        #      actions.escalate         -> (flag / early-stop)
        # 4. event.is_final_response()  -> Response.answer
```

**Two integration modes (pick per use case):**
- **(a) ADK owns its tools.** Register our *emulated* tools as ADK `FunctionTool`s
  so the agent calls them normally; we just observe events. Most realistic.
- **(b) We own the loop.** Use ADK only as the "policy" and route tool calls back
  through our emulators/caches. Maximum determinism/observability.

> **Q:** ADK is **async** (`run_async`, async `create_session`). Bridge via our
> asyncio `orchestration` layer, or expose `async act`? (See §3.1.)
> **Q:** How much ADK session/state do we surface in the Trajectory (planner
> steps, memory, sub-agents)?
> **Comment:** Make ADK an **optional extra** (`kaggle_benchmarks[adk]`) so the
> core stays dependency-light. Pin/verify the ADK API — it moves fast (the
> `dev/` prototypes are already a version or two behind).
> **Idea:** Same adapter interface → `OpenAIAgentsAgent` (SDK already prototyped
> in `dev/agents.py` via a custom `Model`), `LangGraphAgent`, etc. The eval-Agent
> protocol is the contract; frameworks are plugins. **`LiteLlm` also means we can
> point ADK agents at the Kaggle model proxy**, so the same model set is testable
> across frameworks.

---

## 10. Fairness: model rotation & de-duplication

> Captures the "rotate models for fairness, but kill duplicates" note, extended
> to *both* task authoring *and* user simulation.

**Why rotate the *author* model.** If a single model writes every scenario, the
suite inherits that model's blind spots and style, and can hand a **home-field
advantage** to models from the same family (they "think alike"). Drawing each
task's author from a **panel** of models diversifies difficulty and reduces
generator bias.

**Why rotate the *user-simulator* model.** If the simulated user is always model
X, agents get implicitly tuned to X's phrasing/behavior. Rotate the user-sim
model across scenarios (keep it fixed *within* one conversation for coherence).

**Mechanics (draft):**
- **Author panel:** per task, pick a generator model (round-robin or seeded
  random) from a configured panel. Record `author_model` in the scenario
  (provenance).
- **User-sim panel:** per scenario (or per run), assign a user-sim model from a
  panel; record `user_sim_model`.
- **De-duplication** (more models × more prompts ⇒ more near-dups):
  1. *Exact / near-exact:* normalize + hash text.
  2. *Semantic:* embed each scenario (persona + goal + nuance) with
     sentence-transformers; cluster by cosine similarity; drop/merge within a
     threshold. **Reuse the oulipo `calculate_semantic_diversity_score`
     approach.** (Alt: MinHash/SimHash for cheap scale.)
  3. *Borderline pairs:* optional LLM-judge "are these two testing the same
     thing?".
  - Optimize for **coverage** (spread across tags / nuance-types), not raw count.
- **Provenance & reproducibility:** store `author_model`, `user_sim_model`,
  `seed`, `dedup_threshold`, `embedding_model` in suite metadata → the suite's
  fairness is auditable and regenerable.

**Fairness guardrails (proposed):**
- **No self-authoring:** exclude the model-under-test from the author panel for
  its own evaluation (and ideally from judging itself).
- **No self-grading:** rotate/panel the *judge* too, or at least don't let a model
  grade its own trajectory.

> **Q:** Dedup *threshold* — too aggressive kills legit variety; too loose leaves
> near-clones. Tune on a labeled pair set?
> **Q:** How do we weight aggregate scores when tasks came from different authors
> of differing difficulty? (Ties into M8 difficulty calibration.)
> **Idea:** A "diversity report" artifact per suite: cluster sizes, tag coverage,
> author/user-sim distribution, dropped-duplicate count.

---

## 11. Prior art & related work

**Local prototypes (author's scratch, `dev/` — gitignored).** The Examiner /
agentic-eval idea is already partly prototyped; mine these before building:
- **Agent frameworks driven end-to-end:** ADK (`dev/base_adk.py`, `dev/adk.py`)
  and the OpenAI **Agents SDK** (`dev/agents.py`, incl. a custom model +
  `simulate_booking()` faking a trajectory via `chats.send_step`). → validates the
  multi-framework adapter (§9).
- **Hidden-goal / game simulations:** `dev/tasks/{prisoner,rps,puzzle_gader_games}.py`.
- **Current task→multi-model eval flow:** `dev/tasks/eval_emojis.py`
  (`@bm.task` → `llm.send` → assert → `.evaluate(llm=bm.llms.values())`).
- **Judge-panel pattern:** `dev/t_benchmark.py` (`facts_grounding(..., judges=[...])`).

**Internal design docs (source of the vision).** Two internal Google Docs
describe this idea (incl. a "tasks call-to-action"). They're **auth-walled — I
couldn't read them** (HTTP 401 via `webfetch`), so nothing from them is reflected
here yet. Pointers (URLs, owners) are stored in `.kagent-context/INTERNAL.md`
(gitignored), **not here** — this file is external-clean.

> **Q — need from you:** paste/export the two docs (or grant access) so I can fold
> in the real generation/simulation/judging/dedup details.
> **Q:** which of the `dev/` prototypes is closest to the intended Examiner
> flow — should I lift one as the M0/M3 starting point?

---

## 12. Open questions / parking lot

- **Determinism vs realism** for tool emulators and generation. Caching + seeds?
- **Cost.** Generation + simulation + judging is a lot of model calls. Budgeting,
  caching, and "cheap mode" (rules-only analyzers) matter.
- **Trust in the judge.** How do we validate the Examiner's grades? Human
  spot-checks? Judge-vs-judge agreement? Calibration set with known answers?
- **Fairness / leakage.** Frozen suites can leak into training data over time.
  Rotating/hidden test sets? Private suites?
- **Multi-turn user simulators** drifting off-persona or being too helpful/too
  adversarial. Need persona-adherence checks (an analyzer on the *user* too).
- **Where do trajectories live** relative to Runs (embedded vs referenced).
- **Async / concurrency** story for running many agents × many tasks.
- **Naming** of every concept here — `Examiner` and the eval-facing agent (vs the
  existing `Actor`) are the two most load-bearing to settle.

---

## 13. Glossary (keep honest as terms settle)

- **Agent (eval-facing)** — the black box under test: takes a conversation and
  produces an answer + trajectory. *(Distinct from the library's existing `Actor`,
  which is a speaker identity — naming TBD; see M0.)*
- **Trajectory** — the recorded steps an Actor took (messages, tool calls,
  reasoning, handoffs) + final answer + usage.
- **Scenario / Task** — one generated case: persona, goal, shared context, hidden
  nuances, environment ground truth, rubric.
- **Suite** — a frozen, versioned set of scenarios for a domain.
- **Examiner** — the meta-agent that authors scenarios, (optionally) judges
  results, and (v2) calibrates difficulty.
- **Tool emulator** — a fake tool that serves consistent results from a
  scenario's environment (shared + hidden), cacheable.
- **Analyzer** — a check over a trajectory (structural, reasoning, or judge-based)
  that records a result and optionally an error class.

---

<!-- Scratch space below — dump raw thoughts here, promote them up later. -->
## 99. Scratch
- Could a "domain" be shareable like a task on Kaggle? Community-authored suites?
- Diffing two trajectories for the same scenario across models (side-by-side viz).
- "Replay" a stored trajectory against new analyzers without re-running the agent.
- Regression mode: did my agent get *worse* on suite X between v1 and v2?
