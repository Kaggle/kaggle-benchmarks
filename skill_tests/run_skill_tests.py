#!/usr/bin/env python3
"""Automated test harness for the kaggle-benchmarks agent skill file.

For each scenario in ``agent_test_scenarios.md``:

1. Invoke a sub-agent (Anthropic API, configurable model) with ``SKILL.md`` as the
   system prompt and the scenario prompt as the user message.
2. For each criterion in the scenario's Expected Answer checklist, ask a judge
   model (LLM-as-judge) whether the response satisfies the criterion and return
   JSON ``{"passed": bool, "reason": str}``.
3. Optionally execute the generated Python code against ``kaggle_benchmarks``
   (Category 1-3, when ``--execute`` is on).
4. Emit a ``results.json`` (full transcript + per-criterion verdicts) and a
   ``report.md`` with the Summary Table from ``agent_test_scenarios.md`` filled
   in (Strong Pass / Partial Pass / Fail).

Usage examples::

    export ANTHROPIC_API_KEY=sk-ant-...
    python skill_tests/run_skill_tests.py                     # all scenarios
    python skill_tests/run_skill_tests.py --scenarios 1.1,6.1 # subset
    python skill_tests/run_skill_tests.py --category 1        # whole category
    python skill_tests/run_skill_tests.py --parallel 4        # concurrency
    python skill_tests/run_skill_tests.py --dry-run           # parse only
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import datetime
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILL_FILE = REPO_ROOT / "skills" / "kaggle-benchmarks" / "SKILL.md"
SCENARIOS_FILE = REPO_ROOT / "skill_tests" / "agent_test_scenarios.md"

DEFAULT_AGENT_MODEL = "claude-sonnet-4-6"
DEFAULT_JUDGE_MODEL = "claude-sonnet-4-6"

# -------- Code-pattern criteria heuristics ---------------------------------
# A criterion is considered "code-pattern" if it's quoting a literal symbol or
# decorator. The judge is instructed to be strict for these — the literal token
# must appear in extracted code.
CODE_PATTERN_HINT_RE = re.compile(r"`[^`]+`")


# ---------------------------------------------------------------------------
# Scenario parsing
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Scenario:
    id: str               # e.g. "1.1"
    category: int         # e.g. 1
    title: str            # e.g. "Simple Q&A with Regex Check"
    category_title: str   # e.g. "Basic — Simple Task + Assertion (Cookbook Recipes)"
    prompt: str
    criteria: list[str]
    source_of_truth: str  # raw text reference

    @property
    def slug(self) -> str:
        return f"scenario_{self.id.replace('.', '_')}"


SCENARIO_HEADER_RE = re.compile(r"^###\s+Scenario\s+([0-9]+\.[0-9]+)\s+—\s+(.+?)\s*$")
CATEGORY_HEADER_RE = re.compile(r"^##\s+Category\s+([0-9]+):\s*(.+?)\s*$")
SECTION_RE = re.compile(r"^\*\*([A-Za-z][A-Za-z _-]+):\*\*\s*(.*)$")
CHECKBOX_RE = re.compile(r"^\s*-\s*\[\s*[xX ]?\s*\]\s+(.+?)\s*$")
BLOCKQUOTE_LINE_RE = re.compile(r"^>\s?(.*)$")


def parse_scenarios(path: Path) -> list[Scenario]:
    """Parse ``agent_test_scenarios.md`` into structured :class:`Scenario` objects.

    The parser is deliberately tolerant — it scans block by block, tracking the
    current section heading (Prompt / Expected Answer / Source of Truth).
    """
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    scenarios: list[Scenario] = []
    cur_category: int | None = None
    cur_category_title: str = ""
    cur_scn: dict[str, Any] | None = None
    cur_section: str | None = None
    buffer: list[str] = []

    def flush_section() -> None:
        if cur_scn is None or cur_section is None:
            return
        joined = "\n".join(buffer).strip()
        cur_scn[cur_section] = joined

    def finalize_scn() -> None:
        if cur_scn is None:
            return
        flush_section()
        prompt = _extract_blockquote(cur_scn.get("Prompt", ""))
        criteria = _extract_checkboxes(cur_scn.get("Expected Answer", ""))
        scn = Scenario(
            id=cur_scn["id"],
            category=cur_scn["category"],
            title=cur_scn["title"],
            category_title=cur_scn["category_title"],
            prompt=prompt,
            criteria=criteria,
            source_of_truth=cur_scn.get("Source of Truth", "").strip(),
        )
        scenarios.append(scn)

    for raw in lines:
        m = CATEGORY_HEADER_RE.match(raw)
        if m:
            finalize_scn()
            cur_scn = None
            cur_section = None
            buffer = []
            cur_category = int(m.group(1))
            cur_category_title = m.group(2).strip()
            continue

        m = SCENARIO_HEADER_RE.match(raw)
        if m:
            finalize_scn()
            cur_section = None
            buffer = []
            cur_scn = {
                "id": m.group(1),
                "title": m.group(2).strip(),
                "category": cur_category if cur_category is not None else 0,
                "category_title": cur_category_title,
            }
            continue

        if cur_scn is None:
            continue

        m = SECTION_RE.match(raw)
        if m:
            flush_section()
            cur_section = m.group(1).strip()
            inline = m.group(2).strip() if m.group(2) else ""
            buffer = [inline] if inline else []
            continue

        # ``---`` separates scenarios but we already split on next ###.
        if raw.strip().startswith("---") and cur_section is not None:
            # ``---`` outside a section ends data collection for the current scenario
            flush_section()
            cur_section = None
            buffer = []
            continue

        buffer.append(raw)

    finalize_scn()
    return scenarios


def _extract_blockquote(text: str) -> str:
    """Strip leading ``> `` markers used for the Prompt section."""
    out: list[str] = []
    for line in text.splitlines():
        m = BLOCKQUOTE_LINE_RE.match(line)
        if m:
            out.append(m.group(1))
        elif line.strip() == "":
            out.append("")
    # Collapse trailing blank lines and strip
    return "\n".join(out).strip()


def _extract_checkboxes(text: str) -> list[str]:
    out: list[str] = []
    for line in text.splitlines():
        m = CHECKBOX_RE.match(line)
        if m:
            out.append(m.group(1).strip())
    return out


# ---------------------------------------------------------------------------
# Anthropic client wrappers
# ---------------------------------------------------------------------------


def _get_client():
    try:
        import anthropic  # noqa: WPS433  (lazy import: dependency optional)
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "The `anthropic` package is required. Install with: "
            "pip install -r skill_tests/requirements.txt"
        ) from exc
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY to run the harness.")
    return anthropic.Anthropic()


def call_agent(
    client: Any,
    model: str,
    skill_text: str,
    user_prompt: str,
    max_tokens: int = 4096,
) -> str:
    """Run the sub-agent with SKILL.md as the system prompt."""
    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=[
            {
                "type": "text",
                "text": skill_text,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_prompt}],
    )
    return _join_text_blocks(msg.content)


def call_judge(
    client: Any,
    model: str,
    criterion: str,
    response_text: str,
    code_blocks: list[str],
    source_of_truth_excerpt: str,
    scenario_id: str,
    scenario_title: str,
) -> dict[str, Any]:
    """Use an LLM as judge to evaluate a single criterion."""
    extracted = "\n\n".join(f"```python\n{cb}\n```" for cb in code_blocks) or "(no python code blocks found)"
    sot = source_of_truth_excerpt or "(no source-of-truth excerpt available)"

    system = (
        "You are an extremely strict code reviewer evaluating whether an AI agent's "
        "response satisfies a single checklist criterion for a benchmark-writing task. "
        "The criterion may name a specific symbol, decorator, or import that MUST appear "
        "literally in the agent's response (or in the extracted Python code) for the "
        "criterion to pass. Do not be lenient. Do not credit 'close enough' substitutes "
        "if the criterion quotes a specific token in backticks. If the criterion forbids "
        "something ('Does NOT ...'), the absence of the forbidden pattern is required. "
        "Reply ONLY with compact JSON: {\"passed\": true|false, \"reason\": \"...\"}. "
        "Keep the reason under 240 characters."
    )

    user = textwrap.dedent(
        f"""
        Scenario: {scenario_id} — {scenario_title}

        Criterion to evaluate:
        ---
        {criterion}
        ---

        Agent's full response:
        ---
        {response_text}
        ---

        Extracted Python code blocks (if any):
        ---
        {extracted}
        ---

        Source-of-truth excerpt (for ground truth — may be empty):
        ---
        {sot}
        ---

        Return ONLY the JSON object. No commentary.
        """
    ).strip()

    msg = client.messages.create(
        model=model,
        max_tokens=400,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    raw = _join_text_blocks(msg.content).strip()
    return _parse_judge_json(raw)


def _join_text_blocks(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        text = getattr(block, "text", None)
        if text is None and isinstance(block, dict):
            text = block.get("text")
        if text:
            parts.append(text)
    return "".join(parts)


JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_judge_json(raw: str) -> dict[str, Any]:
    """Be defensive: judge may wrap JSON in a code fence."""
    m = JSON_OBJ_RE.search(raw)
    if not m:
        return {"passed": False, "reason": f"Judge returned non-JSON: {raw[:200]!r}"}
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError as exc:
        return {"passed": False, "reason": f"Judge JSON parse error: {exc}; raw={raw[:200]!r}"}
    passed = bool(data.get("passed", False))
    reason = str(data.get("reason", ""))[:500]
    return {"passed": passed, "reason": reason}


# ---------------------------------------------------------------------------
# Code extraction & execution
# ---------------------------------------------------------------------------


PY_CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_python_blocks(text: str) -> list[str]:
    """Pull out fenced ```python``` blocks from a markdown response.

    Falls back to returning the whole text as a single block if no fenced
    Python is found but the text looks like Python (contains ``def `` and
    ``import``). Otherwise returns ``[]``.
    """
    blocks = [m.group(1).strip() for m in PY_CODE_FENCE_RE.finditer(text)]
    blocks = [b for b in blocks if b]
    if blocks:
        return blocks
    if "def " in text and "import" in text:
        return [text.strip()]
    return []


# Path references from the Source of Truth, e.g.
#   ``golden_tests/test_cookbook_examples.py`` lines 149–180
# Allow backticks/punctuation between the path and "lines N–M".
SOT_REF_RE = re.compile(
    r"([\w./_-]+\.(?:py|md))(?:[\s`]*lines?\s+([0-9]+)(?:\s*[–—\-]+\s*([0-9]+))?)?",
)


def read_source_of_truth(text: str, repo_root: Path, max_chars: int = 4000) -> str:
    """Try to read referenced lines from the repo to ground the judge."""
    excerpts: list[str] = []
    seen: set[tuple[str, int, int]] = set()
    for m in SOT_REF_RE.finditer(text):
        rel = m.group(1)
        start = int(m.group(2)) if m.group(2) else 1
        end = int(m.group(3)) if m.group(3) else start + 30
        key = (rel, start, end)
        if key in seen:
            continue
        seen.add(key)
        path = (repo_root / rel).resolve()
        # Sanity: must live under the repo and exist
        try:
            path.relative_to(repo_root)
        except ValueError:
            continue
        if not path.exists() or not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        snippet = "\n".join(lines[max(start - 1, 0) : min(end, len(lines))])
        excerpts.append(f"# {rel} lines {start}-{end}\n{snippet}")
        if sum(len(e) for e in excerpts) > max_chars:
            break
    blob = "\n\n".join(excerpts)
    return blob[:max_chars]


def execute_python(code: str, timeout: int = 60) -> dict[str, Any]:
    """Execute a Python snippet in a subprocess. Returns stdout/stderr/returncode."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp = f.name
    try:
        proc = subprocess.run(
            [sys.executable, tmp],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(REPO_ROOT),
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-4000:],
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": -1,
            "stdout": (exc.stdout or b"").decode("utf-8", "replace")[-4000:]
            if isinstance(exc.stdout, (bytes, bytearray))
            else (exc.stdout or "")[-4000:],
            "stderr": "TIMEOUT",
            "timed_out": True,
        }
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Per-scenario evaluation
# ---------------------------------------------------------------------------


def evaluate_scenario(
    scn: Scenario,
    client: Any,
    skill_text: str,
    agent_model: str,
    judge_model: str,
    execute: bool,
) -> dict[str, Any]:
    """Run a single scenario end-to-end."""
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    error: str | None = None
    response_text = ""
    criterion_results: list[dict[str, Any]] = []
    execution: dict[str, Any] | None = None

    try:
        response_text = call_agent(client, agent_model, skill_text, scn.prompt)
        code_blocks = extract_python_blocks(response_text)
        sot_excerpt = read_source_of_truth(scn.source_of_truth, REPO_ROOT)

        for crit in scn.criteria:
            try:
                verdict = call_judge(
                    client=client,
                    model=judge_model,
                    criterion=crit,
                    response_text=response_text,
                    code_blocks=code_blocks,
                    source_of_truth_excerpt=sot_excerpt,
                    scenario_id=scn.id,
                    scenario_title=scn.title,
                )
            except Exception as exc:  # noqa: BLE001
                verdict = {
                    "passed": False,
                    "reason": f"Judge call failed: {exc.__class__.__name__}: {exc}",
                }
            criterion_results.append({"criterion": crit, **verdict})

        if execute and scn.category in (1, 2, 3) and code_blocks:
            execution = execute_python(code_blocks[0])
    except Exception as exc:  # noqa: BLE001
        error = f"{exc.__class__.__name__}: {exc}\n{traceback.format_exc()[:2000]}"

    passed_count = sum(1 for r in criterion_results if r.get("passed"))
    total = len(criterion_results)
    if total == 0:
        rating = "fail"
    elif passed_count == total:
        rating = "strong_pass"
    elif passed_count >= max(1, total - 1):
        rating = "partial_pass"
    elif passed_count >= total / 2:
        rating = "partial_pass"
    else:
        rating = "fail"

    # If execution is enabled and fails, downgrade strong_pass -> partial_pass
    if execute and execution is not None and rating == "strong_pass":
        if execution.get("returncode", 0) != 0 or execution.get("timed_out"):
            rating = "partial_pass"

    if error is not None:
        rating = "fail"

    return {
        "scenario_id": scn.id,
        "category": scn.category,
        "title": scn.title,
        "started_at": started,
        "prompt": scn.prompt,
        "criteria": scn.criteria,
        "response": response_text,
        "criterion_results": criterion_results,
        "passed_count": passed_count,
        "total_criteria": total,
        "rating": rating,
        "execution": execution,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


RATING_EMOJI = {
    "strong_pass": "✅",
    "partial_pass": "⚠️",
    "fail": "❌",
}


SUMMARY_HEADER = "## Scoring Guide"
SUMMARY_TABLE_HEADER = "### Summary Table"


def render_summary_table(
    scenarios: list[Scenario],
    results_by_id: dict[str, dict[str, Any]],
) -> str:
    """Render the Summary Table from agent_test_scenarios.md filled in."""
    # Pull the original table from the source file so categories/difficulties match.
    src = SCENARIOS_FILE.read_text(encoding="utf-8")
    idx = src.find(SUMMARY_TABLE_HEADER)
    if idx == -1:
        # Fallback: build our own minimal table
        return _build_table_from_scratch(scenarios, results_by_id)
    table_md = src[idx:]

    # Replace each `| {id} | ... | ... | ... | |` (empty trailing Result cell)
    # with the rating emoji + criteria ratio + link to the per-scenario section.
    row_re = re.compile(r"^\|\s*([0-9]+\.[0-9]+)\s*\|(.*?)\|\s*\|\s*$")
    out_lines: list[str] = []
    for line in table_md.splitlines():
        m = row_re.match(line)
        if not m:
            out_lines.append(line)
            continue
        scn_id = m.group(1)
        middle = m.group(2).strip()
        result = results_by_id.get(scn_id)
        if result is None:
            cell = "—"
        else:
            emoji = RATING_EMOJI.get(result["rating"], "?")
            ratio = f"{result['passed_count']}/{result['total_criteria']}"
            cell = f"[{emoji} {ratio}](#scenario-{scn_id.replace('.', '-')})"
        out_lines.append(f"| {scn_id} | {middle} | {cell} |")
    return "\n".join(out_lines)


def _build_table_from_scratch(
    scenarios: list[Scenario],
    results_by_id: dict[str, dict[str, Any]],
) -> str:
    rows = ["| # | Scenario | Category | Result |", "|---|----------|----------|--------|"]
    for scn in scenarios:
        result = results_by_id.get(scn.id)
        if result is None:
            cell = "—"
        else:
            emoji = RATING_EMOJI.get(result["rating"], "?")
            cell = f"{emoji} {result['passed_count']}/{result['total_criteria']}"
        rows.append(f"| {scn.id} | {scn.title} | {scn.category} | {cell} |")
    return "\n".join(rows)


def render_report(
    scenarios: list[Scenario],
    results: list[dict[str, Any]],
    agent_model: str,
    judge_model: str,
    execute: bool,
) -> str:
    results_by_id = {r["scenario_id"]: r for r in results}
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    total = len(results)
    strong = sum(1 for r in results if r["rating"] == "strong_pass")
    partial = sum(1 for r in results if r["rating"] == "partial_pass")
    fail = sum(1 for r in results if r["rating"] == "fail")

    parts: list[str] = []
    parts.append("# kaggle-benchmarks Skill Test Report\n")
    parts.append(f"- Generated: {timestamp}")
    parts.append(f"- Agent model: `{agent_model}`")
    parts.append(f"- Judge model: `{judge_model}`")
    parts.append(f"- Execute generated code: {'yes' if execute else 'no'}")
    parts.append(
        f"- Totals: {strong} Strong Pass · {partial} Partial Pass · {fail} Fail "
        f"({total} scenarios run)\n"
    )

    parts.append(render_summary_table(scenarios, results_by_id))
    parts.append("\n---\n\n## Per-Scenario Detail\n")

    for r in results:
        anchor = f"scenario-{r['scenario_id'].replace('.', '-')}"
        emoji = RATING_EMOJI.get(r["rating"], "?")
        parts.append(f'<a id="{anchor}"></a>')
        parts.append(
            f"### {emoji} Scenario {r['scenario_id']} — {r['title']} "
            f"({r['passed_count']}/{r['total_criteria']})"
        )
        parts.append(f"**Rating:** `{r['rating']}`\n")
        if r.get("error"):
            parts.append(f"**Error:** {r['error']}\n")
        parts.append("**Criteria:**\n")
        for cr in r["criterion_results"]:
            mark = "✅" if cr["passed"] else "❌"
            parts.append(f"- {mark} {cr['criterion']}")
            if cr.get("reason"):
                parts.append(f"    - _judge:_ {cr['reason']}")
        if r.get("execution"):
            ex = r["execution"]
            parts.append(
                f"\n**Execution:** returncode={ex.get('returncode')} "
                f"timed_out={ex.get('timed_out')}"
            )
            if ex.get("stdout"):
                parts.append("```\n" + ex["stdout"] + "\n```")
            if ex.get("stderr"):
                parts.append("_stderr:_\n```\n" + ex["stderr"] + "\n```")
        parts.append("\n<details><summary>Agent response transcript</summary>\n")
        parts.append("\n```\n" + r.get("response", "") + "\n```\n")
        parts.append("</details>\n\n---\n")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def select_scenarios(
    all_scenarios: list[Scenario],
    scenario_ids: list[str] | None,
    categories: list[int] | None,
) -> list[Scenario]:
    if not scenario_ids and not categories:
        return all_scenarios
    out: list[Scenario] = []
    for scn in all_scenarios:
        if scenario_ids and scn.id in scenario_ids:
            out.append(scn)
            continue
        if categories and scn.category in categories:
            out.append(scn)
            continue
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenarios", help="Comma-separated scenario IDs, e.g. 1.1,1.9,6.1")
    p.add_argument("--category", help="Comma-separated category numbers, e.g. 1,2")
    p.add_argument("--agent-model", default=DEFAULT_AGENT_MODEL)
    p.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    p.add_argument(
        "--model",
        help="Convenience: set both agent and judge model to this value",
    )
    p.add_argument("--parallel", type=int, default=1, help="Concurrent scenario workers")
    p.add_argument(
        "--execute",
        action="store_true",
        help="Also execute Cat 1-3 generated code (requires kaggle_benchmarks installed)",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Override results output dir (default: skill_tests/results/<timestamp>)",
    )
    p.add_argument("--dry-run", action="store_true", help="Parse scenarios and exit")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.model:
        args.agent_model = args.model
        args.judge_model = args.model

    scenarios = parse_scenarios(SCENARIOS_FILE)
    print(f"Parsed {len(scenarios)} scenarios from {SCENARIOS_FILE.name}", file=sys.stderr)

    scenario_ids = [s.strip() for s in args.scenarios.split(",")] if args.scenarios else None
    categories = (
        [int(s.strip()) for s in args.category.split(",")] if args.category else None
    )
    selected = select_scenarios(scenarios, scenario_ids, categories)
    print(f"Selected {len(selected)} scenarios to run", file=sys.stderr)

    if args.dry_run:
        for scn in selected:
            print(
                f"  {scn.id:>4}  cat={scn.category}  criteria={len(scn.criteria):>2}  "
                f"{scn.title}"
            )
        return 0

    skill_text = SKILL_FILE.read_text(encoding="utf-8")
    client = _get_client()

    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else REPO_ROOT
        / "skill_tests"
        / "results"
        / datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing results to {out_dir}", file=sys.stderr)

    def _run_one(scn: Scenario) -> dict[str, Any]:
        print(f"  > scenario {scn.id} starting…", file=sys.stderr, flush=True)
        result = evaluate_scenario(
            scn=scn,
            client=client,
            skill_text=skill_text,
            agent_model=args.agent_model,
            judge_model=args.judge_model,
            execute=args.execute,
        )
        print(
            f"  < scenario {scn.id} {RATING_EMOJI.get(result['rating'], '?')} "
            f"{result['passed_count']}/{result['total_criteria']}",
            file=sys.stderr,
            flush=True,
        )
        return result

    results: list[dict[str, Any]] = []
    if args.parallel > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = {pool.submit(_run_one, scn): scn for scn in selected}
            for fut in concurrent.futures.as_completed(futures):
                results.append(fut.result())
        results.sort(key=lambda r: tuple(int(p) for p in r["scenario_id"].split(".")))
    else:
        for scn in selected:
            results.append(_run_one(scn))

    payload = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "agent_model": args.agent_model,
        "judge_model": args.judge_model,
        "execute": args.execute,
        "scenario_count": len(results),
        "results": results,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report = render_report(scenarios, results, args.agent_model, args.judge_model, args.execute)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    print(f"Wrote {out_dir / 'results.json'} and {out_dir / 'report.md'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
