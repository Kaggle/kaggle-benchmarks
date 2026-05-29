# Skill Test Harness for `kaggle-benchmarks`

Automated harness that grades the [`skills/kaggle-benchmarks/SKILL.md`](../skills/kaggle-benchmarks/SKILL.md) instruction file against the scenarios defined in [`agent_test_scenarios.md`](./agent_test_scenarios.md).

For each scenario the harness:

1. Loads `SKILL.md` as the **system prompt** for a sub-agent (Anthropic API, Claude Sonnet 4.6 by default).
2. Sends the scenario's `Prompt` as the **user message** and captures the response.
3. Uses an **LLM-as-judge** (also configurable) to score the response against every checkbox criterion. The judge receives:
    - the criterion text,
    - the agent's full response,
    - the extracted Python code blocks (so code-pattern criteria are checked against code, not prose),
    - and an excerpt of the cited Source-of-Truth file (when the SoT cites `path/to/file.py lines N–M`).
4. Optionally **executes** the generated Python (`--execute`) against the local `kaggle_benchmarks` install. Cat 1–3 only.
5. Writes `results.json` (full transcript + per-criterion verdicts) and `report.md` (the Summary Table from `agent_test_scenarios.md` filled in with ✅ / ⚠️ / ❌, plus per-scenario detail).

## Requirements

- Python ≥ 3.11 (matches the repo).
- `pip install -r skill_tests/requirements.txt` (just the `anthropic` SDK).
- `ANTHROPIC_API_KEY` in your environment.
- Optional, for `--execute`:
  - `pip install -e .` (or `uv pip install -e .`) from the repo root to install `kaggle_benchmarks`.
  - The same env vars `kaggle_benchmarks` needs to talk to its LLM backend (see [`.env.example`](../.env.example)).

## Usage

```bash
export ANTHROPIC_API_KEY=sk-ant-...

# Run all 54 scenarios sequentially (cheapest, slowest)
python skill_tests/run_skill_tests.py

# Run a subset
python skill_tests/run_skill_tests.py --scenarios 1.1,1.9,6.1

# Run a whole category
python skill_tests/run_skill_tests.py --category 6

# Run with concurrency (each scenario runs independently)
python skill_tests/run_skill_tests.py --parallel 4

# Pick the agent and judge models independently
python skill_tests/run_skill_tests.py \
    --agent-model claude-sonnet-4-6 \
    --judge-model claude-opus-4-7

# Or set both at once
python skill_tests/run_skill_tests.py --model claude-sonnet-4-6

# Also execute Cat 1-3 generated code (requires `pip install -e .`)
python skill_tests/run_skill_tests.py --category 1 --execute

# Parse scenarios only — no API calls (useful for debugging the parser)
python skill_tests/run_skill_tests.py --dry-run
```

Results land in `skill_tests/results/<UTC-timestamp>/`:

```
skill_tests/results/20260529T214500Z/
├── results.json    # Full transcript + per-criterion verdicts
└── report.md       # Summary table + per-scenario detail (open in any markdown viewer)
```

Override the output directory with `--output-dir path/to/dir`.

## Rating rubric

Per [the scoring guide in `agent_test_scenarios.md`](./agent_test_scenarios.md#scoring-guide):

| Rating | Emoji | When it's awarded |
|--------|-------|-------------------|
| Strong Pass | ✅ | All criteria for the scenario passed the judge. |
| Partial Pass | ⚠️ | Most criteria passed (at least half, but not all). Minor omissions. |
| Fail | ❌ | More than half of the criteria failed, or the agent/judge call raised an exception. |

If `--execute` is enabled and the generated code crashes (non-zero return code or timeout), a Strong Pass is downgraded to Partial Pass — the agent wrote something that looked right but didn't actually work.

The judge is instructed to be **strict**: a criterion like `Uses @kbench.task() decorator` only passes if the literal `@kbench.task()` (or a documented variant such as `@kbench.task(name="…")`) appears in the agent's response or extracted Python code. Negative criteria (`Does NOT use plain assert`) require the absence of the forbidden token.

## Smoke test

`skill_tests/smoke.sh` runs a 3-scenario subset (1.1, 1.9, 6.1) covering a simple task, the simplest possible task, and a knowledge question:

```bash
ANTHROPIC_API_KEY=sk-... bash skill_tests/smoke.sh
```

This is what CI should call to confirm the harness itself still works without paying for all 54 scenarios.

## Design notes

- **Criteria are parsed live** from `agent_test_scenarios.md`. Add or edit a checkbox there and the next harness run picks it up — no code changes needed.
- **Code extraction** strips ```` ```python ```` fences before handing to the judge for code-pattern criteria. If no fenced block is present, the whole response is treated as a single block (some scenarios are knowledge questions with no code).
- **Source-of-truth excerpts** are pulled from the repo using the `path lines N–M` references in each scenario, then included in the judge prompt to ground its evaluation.
- The harness is **model-agnostic** via `anthropic.Anthropic()`. Pass any model name your account has access to via `--agent-model` / `--judge-model`.
- **Concurrency** uses a thread pool — the Anthropic SDK releases the GIL on network calls, so threads give meaningful speedup without process overhead.
- **Prompt caching** is enabled on the SKILL.md system prompt (the same ~1200-line file is sent on every scenario), so a multi-scenario run hits the cache after the first call.
