# Positioning agentic-eval: two value propositions

> Status: DRAFT proposal — companion to `design/agentic-evaluation.md`.
> Purpose: separate agentic-eval's two distinct value propositions and expand
> on the second (agentic-eval as a framework-agnostic analysis layer), which is
> currently underdeveloped in the main design doc.

## 0. The framing

Agentic-eval as currently designed bundles two things that don't have to be
bundled:

1. **A native runtime** for cheap, in-process behavioral evaluation on
   hand-authored or generated scenarios (fake dict worlds, emulated tools,
   `simulate()`, the Examiner).
2. **A trajectory analysis stack** — a typed `Trajectory`, an analyzer library
   (`called_tool`, `reasoning_mentions`, `judge`, ...), an error taxonomy
   (`ErrorClass`), and `Report` aggregation.

Value proposition **(1)** shines on domains where success is behavioral, ground
truth fits in a dict, and iteration speed matters (legal drafting, customer
support, medical triage, advisory tasks, harness/tool ablations). Value
proposition **(2)** is portable: it can run over trajectories produced by
*any* runtime — Harbor, ADK, OpenAI Agents SDK, LangGraph, custom loops.

The main design doc treats (1) as the headline and (2) as an internal
implementation detail. This proposal argues (2) is the higher-leverage,
more broadly useful contribution and deserves to be first-class.

---

## 1. Value proposition (1) — recap

Covered thoroughly in `design/agentic-evaluation.md`. Summary of the fit
criteria for using agentic-eval's *native runtime*:

- Structured world truth expressible as a dict (`Scenario.environment`).
- No objective machine-checker for the final output — success is judged by
  process, not artifact.
- Failure modes are diverse enough that a taxonomy (`ErrorClass`) yields more
  signal than a scalar.
- Iteration speed matters (per-checkpoint diagnosis, harness ablations).
- Coverage via generation matters — a paragraph → 25 scenarios via the
  Examiner beats hand-authoring.

Poor fit: real filesystems, real containers, artifact-checkable outputs
(SWE-Bench, Terminal-Bench, math proofs). Those belong in Harbor.

**This value proposition is real and worth keeping.** The rest of this doc is
about the second one.

---

## 2. Value proposition (2) — agentic-eval as a framework-agnostic analysis layer

### 2.1 The core claim

The analysis stack — trajectory type + accessors + analyzer library + error
taxonomy + reports — is decoupled from *how the trajectory was produced*. Any
runtime that can emit an event stream (Harbor's ATIF, ADK's Event stream,
OpenAI Agents SDK, custom loops) can be adapted into an agentic-eval
`Trajectory` via a small format bridge (~30–50 lines each, same shape as the
existing ADK adapter in `examples/agentic_eval/05_adk_adapter.py`).

Once adapted, the same analyzers, taxonomy, judges, and reports apply — giving
users a **shared analytical vocabulary across frameworks and runtimes**.

### 2.2 Why this is the higher-leverage piece

Enumerating the arguments:

**A. It multiplies leverage per unit of code.**
The analysis stack is written once. Every new format bridge (~30–50 lines) is
amortized across every future trajectory in that format. Every new analyzer is
amortized across every runtime that has a bridge. The math is multiplicative.

**B. It addresses a real gap in the ecosystem.**
Harbor gives you a scalar reward. ADK gives you an event log. OpenAI Agents
SDK gives you a run object. None of them ship a decomposition of *why* a run
failed. Everyone rolling their own analysis re-invents:
- Trajectory walking (get me the tool calls, get me the reasoning).
- A taxonomy vocabulary (what do we call "the agent didn't use the tool it
  should have"?).
- A judge protocol (structured question → verdict → reason).
- Report aggregation (per-run, per-suite, per-agent, per-error-class).

Agentic-eval already has all of these.

**C. It's the piece both major user groups need.**

- *Model trainers* need decomposed failure categories to diagnose training
  runs and drive next-experiment decisions.
- *Model buyers / users* need re-projectable data to compare models on
  behaviors that matter for their use case, not the leaderboard's aggregate.

Both need the analysis layer. Neither strictly needs the native runtime.

**D. The runtime is not the moat.**
Harbor's runtime is production-hardened at cloud scale; agentic-eval's
runtime is intentionally notebook-scale. Trying to compete on the runtime
axis is a losing proposition. Trying to complement Harbor with a better
analysis story is a winning one.

**E. It stays true to the current design.**
Design doc §1: "Not a production agent runtime." Positioning agentic-eval's
analysis stack as portable acknowledges this constraint and turns it into a
strength — the stack goes wherever trajectories are, rather than trying to
own the runtime.

### 2.3 Architecture — the stack, layered

The current codebase already has these layers; they're just not marketed as
separable. Making them explicit:

```
┌─────────────────────────────────────────────────────────┐
│  Reports / Leaderboards / Aggregation                   │  <- consumer-facing
│  (Report, Examiner.grade, per-error-class rollups)      │
├─────────────────────────────────────────────────────────┤
│  Analyzers + Error Taxonomy                             │  <- reusable checks
│  (called_tool, reasoning_mentions, judge, ErrorClass)   │
├─────────────────────────────────────────────────────────┤
│  Trajectory Model                                       │  <- canonical form
│  (Trajectory, Message, LLMMessage, ToolInvocation,      │
│   ToolInvocationResult, accessors, __panel__)           │
├─────────────────────────────────────────────────────────┤
│  Format Bridges (adapters)                              │  <- new/expanded
│  ┌───────┬─────┬───────────────┬──────────┬──────────┐  │
│  │ ATIF  │ ADK │ OpenAI Agents │ LangGraph│ custom   │  │
│  └───────┴─────┴───────────────┴──────────┴──────────┘  │
├─────────────────────────────────────────────────────────┤
│  Runtimes producing trajectories                        │  <- external
│  ┌────────┬───────────┬───────────────┬──────────────┐  │
│  │ Harbor │ ADK live  │ agentic-eval  │ user's own   │  │
│  │ (real) │ (Gemini)  │ simulate()    │ agent loop   │  │
│  └────────┴───────────┴───────────────┴──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

The bottom two layers are pluggable. Everything above is shared. That's the
composition claim.

### 2.4 The bridges — what's built, what's missing

**Built today:**
- ADK (fake stream shown in example 05; live version in example 08).
- Agentic-eval's own runtime (`simulate()` produces Trajectory directly).

**Immediately valuable to build:**
- **Harbor / ATIF** — the highest-leverage bridge. ATIF is well-specified
  (`harbor/src/harbor/models/trajectories/step.py`), rich (per-step timestamps,
  model names, LLM call counts, multimodal parts, sub-agent references), and
  Harbor is the biggest source of trajectories in the ecosystem.
- **OpenAI Agents SDK** — validated shape (see `dev/agents.py` in the
  agentic-evaluation design doc's §11 prior art).

**Deferred / opportunistic:**
- LangGraph, LlamaIndex Agents, LangChain AgentExecutor — same pattern.
- Anthropic's tool-use / MCP conversation records.

Each bridge is a ~30–50 line adapter. Reference implementation:
`examples/agentic_eval/05_adk_adapter.py` (translates ADK events to
`Trajectory.from_steps` tuples).

### 2.5 What ATIF has that Trajectory doesn't (and what to do about it)

Honest inventory. ATIF's `Step` carries several fields agentic-eval's
`Trajectory` currently doesn't:

| ATIF field | Agentic-eval status | Recommendation |
|---|---|---|
| `step_id`, `timestamp` | absent | Add optional `metadata: dict` per step; the bridge populates from ATIF. Non-breaking. |
| `model_name` per step | absent (only on trajectory metadata) | Same — metadata dict. |
| `reasoning_effort` | absent | Metadata. |
| `llm_call_count` | absent | Metadata. |
| `is_copied_context` | absent | Metadata + a `Trajectory.exclude_copied()` view for analyzers. |
| Multimodal `ContentPart[]` in messages | `Message`/`LLMMessage` supports it via existing library types | Verify the bridge unpacks correctly; likely OK. |
| Sub-agent trajectory references (external files) | Nested `Trajectory` as a step | Bridge dereferences the reference and inlines a nested `Trajectory`. |
| `Metrics` (per-step LLM operational data) | Only aggregate `Usage()` on the trajectory | Extend `Usage` to be attachable per-step, or use metadata. |

**Design principle:** don't fork the format. Extend `Trajectory` with an
optional per-step metadata bag and let bridges populate it. Analyzers that
don't need those fields ignore them; analyzers that do can opt in.

### 2.6 The taxonomy question

The current `ErrorClass` (`MISSED_HIDDEN_CONSTRAINT`, `TOOL_UNDERUSE`,
`TOOL_OVERUSE`, `WRONG_TOOL`, `HALLUCINATED_FACT`, `IGNORED_USER_GOAL`,
`UNSAFE_ACTION`, `GAVE_UP_EARLY`, `FORMAT_VIOLATION`) is deliberately a
strawman (§3.3 of the main design doc). Positioning agentic-eval as an
analysis layer for external runtimes puts pressure on the taxonomy:

- **Generic classes stay generic.** The nine above cover a lot; keep them.
- **Domain-specific classes are user-extended.** A coding-agent user might
  add `PATCHED_WITHOUT_REPRO`, `IGNORED_FAILING_TEST`. A medical-agent user
  might add `MISSED_RED_FLAG`, `PREMATURE_CLOSURE`. The taxonomy should be an
  open enumeration — `ErrorClass` is a namespace of *conventional* labels,
  not a closed set.
- **Cross-runtime consistency matters.** If a Harbor run and an ADK run both
  produce `TOOL_UNDERUSE`, users benefit from the shared meaning. A style
  guide for what each label means keeps consistency; auto-generated docs
  from analyzer implementations would help.

### 2.7 Consumption patterns unlocked

Once the analysis stack is portable, three consumption patterns emerge:

**Pattern P1: Re-analyze after the fact.**
Team ran Harbor last week. Someone realizes they care about a new behavior.
Convert existing ATIF trajectories → `Trajectory`, run the new analyzer, get
the answer without re-running any inference. Zero-cost retroactive analysis.

**Pattern P2: Multi-runtime dashboards.**
A model is evaluated in both Harbor (real environments) and agentic-eval's
own runtime (behavioral probes). Both produce `Report`s in the same shape.
One dashboard, one leaderboard, one error-class distribution — across
runtimes.

**Pattern P3: Framework-agnostic benchmark suites.**
An analyzer suite (`analyzers_for_coding_agent()`, returning a list of
`Analyzer`s) becomes a reusable artifact you can point at any trajectory
from any runtime. Users can share and publish analyzer suites the way they
share benchmark datasets today.

None of these are possible today. All become possible with format bridges +
the existing analysis stack.

---

## 3. Concrete proposal

### 3.1 Repositioning

Update the main design doc's framing (§0 Elevator pitch) from:

> **Two products in one, sharing a spine:**
> 1. End-to-end, batteries-included evaluation.
> 2. A low-level toolkit.

To:

> **Three products, sharing a spine:**
> 1. **The analysis stack** — a framework-agnostic trajectory type, analyzer
>    library, error taxonomy, and reporting layer, usable over trajectories
>    from any runtime (Harbor, ADK, OpenAI Agents SDK, custom loops).
> 2. **The behavioral evaluation runtime** — cheap in-process
>    `simulate()`-driven evaluation on scenarios where success is behavioral
>    and ground truth fits in a dict.
> 3. **The scenario generator (Examiner)** — bootstrap a suite of scenarios
>    from a paragraph describing a domain.

Each of the three products is independently valuable and independently
adoptable. A user might adopt only (1) to analyze their Harbor runs; another
might adopt only (2) for cheap behavioral probes; a third might use all three
end-to-end.

### 3.2 Concrete work items (proposed prioritization)

**W1. ATIF bridge (highest leverage).**
New module `kaggle_benchmarks.agentic.harbor` (or `agentic.bridges.atif`).
One function `trajectory_from_atif(events) -> Trajectory`, tested against
Harbor's ATIF fixtures. Size: ~50 lines + tests. Unlocks all of Section 2.7.

**W2. Optional per-step metadata bag on `Trajectory`.**
Extend the trajectory step types with an optional `metadata: dict` field so
bridges can attach timestamps, model names, `llm_call_count`, etc. without
requiring analyzers to care. Non-breaking. Size: small, mostly plumbing.

**W3. Analyzer namespacing + extension guide.**
Formalize the `ErrorClass` namespace as extensible (users add domain classes
in their own modules). Publish a short guide on writing analyzers that
produce those classes. Size: docs + light refactor.

**W4. Reference "shared vocabulary" analyzer suite for coding agents.**
Prove the pattern with one concrete domain: analyzers that make sense for
SWE-Bench-style coding trajectories (`read_failing_test_first`,
`reproduced_before_patching`, `used_search_before_edit`, etc.). Ships as
`kaggle_benchmarks.agentic.analyzers.coding`. Bridge + suite together let a
user run Harbor SWE-Bench today and get a categorized behavioral report.

**W5. OpenAI Agents SDK bridge.**
Second bridge to prove the pattern isn't ADK-specific. Validates the design
of the metadata bag against a second real format. Size: ~30–50 lines +
tests.

**W6. Multi-runtime report aggregation.**
Extend `Report` / aggregation helpers to carry a `source_runtime` label so
dashboards can slice per-runtime. Size: small, cosmetic + one field.

**W7. Format-bridge authoring guide.**
Document the pattern (event stream → `Trajectory.from_steps` tuples → metadata
attach). Reference the ADK adapter as canonical. Enables community
contributions of bridges for LangGraph, LlamaIndex, etc.

### 3.3 What this proposal does *not* propose

For honesty and scoping:

- **Not proposing** that agentic-eval replace Harbor or grow a cloud
  environment layer. That would fight the design doc's explicit non-goal
  ("Not a production agent runtime") and duplicate what Harbor does well.
- **Not proposing** a general "Harbor task → agentic-eval scenario"
  converter. That mapping is category-dependent and mostly not
  representable as an automated transform (see prior discussion on
  environment shape mismatch).
- **Not proposing** to change `simulate()`, `EmulatedTool`, or the Examiner
  in any way. Value proposition (1) stands unchanged; this proposal only
  factors out and elevates (2).

---

## 4. Success criteria for the analysis-layer positioning

If this repositioning works, we should see:

- **Users of Harbor** who don't otherwise touch agentic-eval import the
  bridge + analyzer library, and generate categorized reports over their
  existing runs.
- **A published analyzer suite** for at least one benchmark domain (coding
  agents; medical; legal) that's actively used and extended.
- **At least one third-party bridge** contributed by the community (LangGraph
  or similar), demonstrating the pattern generalizes.
- **A single leaderboard** in some team's workflow that combines Harbor runs
  and agentic-eval native runs, keyed by the same error-class taxonomy.

If none of these materialize within a reasonable window, the analysis layer
isn't as portable/valuable as this proposal claims, and we should
consolidate back around (1) + (3).

---

## 5. Open questions

- **Storage / interop with existing benchmark protos.** Where do converted
  `Trajectory` objects persist? Reuse `*.run.json` proto? Introduce a
  `*.trajectory.json`? Depends on §4.5 of the main design doc's storage
  backend decisions.
- **Taxonomy governance.** If domain-specific `ErrorClass` values proliferate
  across teams, do we curate a shared registry or let them stay local? What's
  the cost of a fragmented taxonomy?
- **Bridge fidelity testing.** How do we validate that a bridge is
  loss-preserving? Round-trip tests where possible; property tests on
  synthetic ATIF; golden fixtures from real Harbor runs.
- **Analyzer trust across runtimes.** An analyzer written against
  agentic-eval's `simulate()` trajectories may behave differently on ATIF
  ones (e.g., subtle differences in what counts as a "reasoning" step).
  Need a conformance test suite that runs the same analyzers against
  hand-authored trajectories in each source format and checks they produce
  consistent results.
- **What about non-trajectory signal?** Harbor also produces reward, exit
  codes, artifacts. Should `Report` be extended to carry these alongside
  analyzer results, or should they stay in Harbor's own report format and
  be joined externally?

---

## 6. TL;DR

Agentic-eval currently bundles a runtime and an analysis stack. The runtime
is useful for a specific fit (§1). The analysis stack is useful for a much
broader audience — anyone who runs an agent through *any* framework and
wants structured behavioral decomposition of what happened.

This proposal argues the analysis stack should be positioned as a
first-class, framework-agnostic product, with format bridges (ATIF, ADK,
OpenAI Agents SDK, ...) as the composition mechanism. Concrete first step:
build the ATIF bridge and ship a reference analyzer suite for coding agents,
so Harbor users can turn scalar rewards into typed behavioral reports without
adopting anything else in agentic-eval.
