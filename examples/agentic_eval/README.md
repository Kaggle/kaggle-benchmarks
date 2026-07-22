# Agentic evaluation — demos

Runnable demos for the vision in
[`design/agentic-evaluation.md`](../../design/agentic-evaluation.md).

The implementation now lives in the library as the **experimental
`kaggle_benchmarks.agentic`** subpackage, built on the library's own custom
types (`Chat` / `Message` / `LLMMessage` / `Usage` / `ToolInvocation` /
`AssertionResult`). LLMs/agents/tools are **mocked** (`kaggle_benchmarks.agentic.demo`)
so the flows are deterministic.

> Status: 🛩️ **experimental.** The backbone (trajectory, analyzers,
> scenarios/suites, simulation, examiner, rotation/dedup, **visualization**) is
> implemented on real types; the model/agent parts are mocked. Not wired into the
> package's top-level API — import it explicitly.

## Run

Needs an installed `kaggle_benchmarks` (e.g. `uv pip install -e .`):

```bash
cd examples/agentic_eval
jupytext --to notebook 03_simulate_travel.py && jupyter lab 03_simulate_travel.ipynb
```

These are **jupytext percent-format notebooks** (`# %%` / `# %% [markdown]`
cells), matching `documentation/examples/`. Open them directly in Jupyter/VS Code,
or `jupytext --to notebook <file>.py`. They also run as plain scripts
(`python 03_simulate_travel.py`), though the `__panel__` visualizations only show
in a notebook.

## What each demo shows

| Notebook | Design doc | Shows |
|---|---|---|
| `00_render_gallery.py` | §3–4 | **no agents/LLMs** — hand-build (or load from JSON) a `Trajectory`, `Scenario`, `Suite`, `Report` and just render them |
| `01_trajectory_and_analyzers.py` | §3.2, §3.3 | build a `Trajectory` (from `Message`/`LLMMessage`/`ToolInvocation`), render it, run analyzers (which return `AssertionResult`) |
| `02_scenario_and_examiner.py` | §4.1, §10 | Examiner authors a Suite with **author-model rotation** + **near-duplicate pruning**, provenance, save/load |
| `03_simulate_travel.py` | §4.2 | the travel scenario end-to-end: simulated user + env-aware **emulated tools** (hidden football match) + two agents |
| `04_evaluate_and_report.py` | §4.3 | grade both agents, **leaderboard** + **error taxonomy** (lazy agent → `missed_hidden_constraint`) |
| `05_adk_adapter.py` | §9 | map a (fake) **ADK** event stream into a `Trajectory` via the `from_steps` constructor |
| `06_llm_agent_and_mapping.py` | §7 | mock the LLM against the real `LLMChat`, drive `LLMAgent`; concept → primitive mapping |

## Where the code lives: `kaggle_benchmarks.agentic`

| Module | Concept |
|---|---|
| `trajectory.py` | `Trajectory` over `Message`/`LLMMessage`/`ToolInvocation`/`ToolInvocationResult`; `from_chat` / `from_steps`; `__panel__` viz |
| `agent.py` | eval-facing `Agent`: `LLMAgent` (real, wraps `LLMChat` + `native_tool_agent`), `PlannedAgent` (scripted), `ConstantAgent` |
| `analyzers.py` | analyzers returning `AssertionResult` + error taxonomy (`ErrorClass`) |
| `scenario.py` | `Scenario` + `Suite` (content-hash version, JSON save/load, `__panel__`); `Persona` is a subclass of `actors.Actor` (it's a first-class speaker with a profile + goal) |
| `simulation.py` | `simulate()`, `EmulatedTool` (env-aware, cached), `UserSimulator` |
| `examiner.py` | `Examiner` (author + grade), `Report` (`__panel__`) |
| `fairness.py` | model rotation (`pick`), `dedup`, `diversity_report` |
| `demo.py` | travel scenario, emulated tools, thorough/lazy agents, keyword judge |

## Visualization

`Trajectory`, `Scenario`, `Suite`, and `Report` implement `__panel__`, so they
render in a notebook (`Message` steps go through the rendering seam;
`AssertionResult` renders itself). Text fallbacks: `Trajectory.render()`.

## What's real vs mocked

- **On real types.** Steps are `Message`/`LLMMessage`/`ToolInvocation`; analyzers
  return `AssertionResult`; usage is `Usage`. Nothing is re-invented.
- **Mocked:** the LLMs (scripted), agents (`PlannedAgent` fixed plans), tool
  results (from `scenario.environment`), the judge (keyword), authoring (canned
  variants), and ADK (a fake event stream).

## The `harness/` folder

`harness/` is a **deprecated compatibility shim** re-exporting
`kaggle_benchmarks.agentic` (the name also collides with a third-party package).
Prefer importing from `kaggle_benchmarks.agentic` directly.

## De-dup note

`fairness.dedup` uses `difflib` string similarity (zero deps). The real version
should embed scenarios (sentence-transformers) and cluster by cosine similarity —
see `documentation/guides/oulipo.py:calculate_semantic_diversity_score`.
