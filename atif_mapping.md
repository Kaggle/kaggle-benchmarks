# Mapping kbench Runs to ATIF

How a `*.run.json` becomes the trajectory and result files harbor tooling reads — the general rules first, then the shapes that do not map one to one.

The converter is `src/kaggle_benchmarks/kaggle/atif.py`. This document describes what it does today, not what it might do.

---

## 1. What Gets Written

Every stored run produces three files, side by side in the same directory:

| File | Holds | Read by |
|------|-------|---------|
| `Foo.run.json` | The run itself. **The only input.** | kbench |
| `Foo.atif.json` | The **trajectory** — the transcript, step by step. | harbor |
| `Foo.result.json` | The **`TrialResult`** — the score, the cost, how it ended. | harbor |

ATIF is harbor's trajectory format. It has no score field, which is why the result is a second file.

The pair is written by `atif.write_beside()` after the run.json is saved. Set `WRITE_ATIF=False` to skip it; the run.json is unaffected either way.

We emit **`ATIF-v1.7`**. That matters in exactly one place — see [§4.7](#47-media-atif-cannot-hold).

---

## 2. Six General Rules

These explain most of what you will see in an output file.

### Absent, never null

Empty values are removed on the way out. A trajectory has no `"cost_usd": null` and no `"tool_calls": []` — those keys are simply not there.

So when reading a converted file, test with `in`, not with a default:

```python
# ❌ DON'T — the key is missing, not null
if step["metrics"]["cost_usd"] is not None: ...

# ✅ DO
if "cost_usd" in step.get("metrics", {}): ...
```

One exception: a tool call always keeps `arguments`, even when the call took none, because harbor requires the key.

### Degrade, never raise

A field the converter cannot map is skipped, logged, and recorded in the file. Conversion failing must never cost someone a finished run.

Every skip appends to `extra.kbench.conversion_warnings`:

```json
"conversion_warnings": [
  {"kind": "media_unrepresentable", "detail": "image/png: inline data"},
  {"kind": "reward_leaf_unusable", "detail": "gold is a str"}
]
```

There are nine kinds, and that is the whole list:

| `kind` | What was skipped, and what happened instead |
|--------|---------------------------------------------|
| `media_unrepresentable` | Media ATIF cannot point at. A text marker took its place — [§4.7](#47-media-atif-cannot-hold). |
| `reward_leaf_unusable` | A score leaf that was not a number. Left out of `rewards`. |
| `result_entry_unprefixed` | A result entry that is neither the headline nor a known split. Its keys went in unprefixed, so two of them collide. |
| `subrun_path_unknown` | A dataset row's trajectory file could not be found. Left out of the refs — [§4.5](#45-dataset-evaluation). |
| `tool_arguments_unparsed` | kbench refused a call and quoted the arguments back as text. `arguments` is empty; the text survives in the result. |
| `tool_result_without_call` | A tool result with no agent step before it to fold onto. Kept as a plain `system` step. |
| `fork_without_common_prefix` | An old-format tool loop that shares no opening with the main transcript. Left as a side chat rather than spliced in — [§4.2](#42-tool-loops). |
| `transcript_truncated_by_serializer` | A tool loop ran out of rounds, so kbench kept nothing past its last closing reply. |
| `unknown_content_role` | A role this converter does not know. Treated as `system`. |

If the list is absent, nothing was skipped. Only one input is fatal: a run.json with no `taskVersion.name` raises `ConversionError`, because harbor requires a task name and there is nothing to fall back to.

### Nothing is dropped silently

A kbench field with no ATIF home is copied verbatim into `extra.kbench`, under its `.proto` name in snake_case. `versionNumber` becomes `version_number`. Fields a newer kbench adds come across too, without a code change.

That is how **assertions** travel. Harbor's verifier is a test script and has no per-step equivalent, so the list is carried whole under `extra.kbench.assertions` and deliberately *not* spread onto the steps it names:

```json
"assertions": [
  {"definition": "assertions.assert_in(\"Berlin\", str(answer))",
   "expectation": "Expected 'Berlin' in 'Paris'",
   "status": "BENCHMARK_TASK_RUN_ASSERTION_STATUS_FAILED",
   "line_number": 0,
   "conversation_request_ids": [
     {"conversation_id": "Checked-0cad7c84", "request_id": "Checked-0cad7c84-req-1"}]}
]
```

Each entry keeps `conversation_request_ids`, so which turn an assertion looked at is still recoverable — but requests are flattened away in the transcript, so it does not resolve to a step number.

The only field deliberately discarded is `Conversation.modelVersionSlug`, which holds a deprecated placeholder string.

### Cost is reported by whoever is scored

Totals in `final_metrics` are summed off the steps that are in this file — the main transcript plus every embedded subagent. They never include subruns, because each subrun writes its own `result.json` and would otherwise be billed twice.

One unpriced message makes the whole bill unknown: `total_cost_usd` is then absent rather than a partial sum. Unmeasured is not the same as free.

### Two protobuf traps

- **A zero scalar disappears.** proto3 omits it, so `"numericResult": {}` means a score of `0.0`, not a missing score.
- **`bool` is an `int` in Python.** A `dictResult` holding `{"is_correct": true}` converts to `{"is_correct": 1.0}` without a warning. That is intended.

### Some values are invented, and look it

Harbor requires fields kbench has no answer for. Rather than guess plausibly, we make them obviously fake, so nobody mistakes a converted run for a real harbor trial:

| Field | Value |
|-------|-------|
| `trial_uri` | `kbench://not-a-harbor-trial` |
| `task_id.path`, `config.task.path` | `kbench://not-a-harbor-task` |
| `task_checksum` | `unknown (kbench)` |
| `agent.name` | `kaggle-benchmarks` — the **harness**, not the model |

Steps kbench added, which nobody said, are prefixed: `[placeholder]`, `[error]`, `[delegated]`.

---

## 3. Field Mapping

### The run

| run.json | ATIF | Notes |
|----------|------|-------|
| `pyRunId` | `session_id`, `trajectory_id`, `trial_name` | Falls back to `id`, then `"unknown"` |
| `taskVersion.name` | `task_name` | Required |
| `modelVersion.slug` | `agent.model_name`, `agent_info.model_info.name` | |
| installed kbench version | `agent.version` | |
| `startTime` / `endTime` | `started_at` / `finished_at` | Already ISO 8601 |
| `errorMessage` | `exception_info` | See [§4.8](#48-failures) |
| `results[]` | `verifier_result.rewards` | See [Scores](#scores) |
| `subruns[]` | `extra.kbench.subruns` + sibling refs | See [§4.5](#45-dataset-evaluation) |
| everything else | `extra.kbench.*` | snake_cased |

Six top-level fields are considered spent once mapped and are not copied on: `conversations`, `results`, `modelVersion`, `pyRunId`, `id`, `startTime`. Everything else is copied whether or not it also has an ATIF home, so `end_time`, `error_message` and `task_version` appear in `extra.kbench` as well — the copy is the complete one, holding the task's `definition` and `description` too. (`subruns` is the exception among the six: it is replaced by a per-row summary rather than dropped.)

Two fields are the converter's own rather than kbench's. `schema_version` is the ATIF version, on the trajectory and on every embedded subagent. `notes` names the source in prose — and when there were no model turns, says why `total_steps` is `0`:

```json
"notes": "Converted from kbench run.json for task 'Room'. No model turn was
          recorded, so total_steps is 0 and the first step below is a placeholder."
```

The trajectory's `final_metrics` are summed off its steps, and `result.json` copies them rather than recomputing, so the two files cannot come to disagree:

| `final_metrics` | `result.json` |
|-----------------|---------------|
| `total_prompt_tokens` | `agent_result.n_input_tokens` |
| `total_completion_tokens` | `agent_result.n_output_tokens` |
| `total_cost_usd` | `agent_result.cost_usd` |
| `total_steps` | — |
| `extra.kbench_result` | `verifier_result.rewards` |

### The transcript

One `Content` becomes one step. Requests are a serializer grouping and are flattened away, except that a request's `metrics` and `reasoningTraces` ride on the assistant reply that closed it — the only step allowed to carry either. A request that does not end in an assistant reply loses both, since there is no turn they belong to.

| run.json | ATIF | Notes |
|----------|------|-------|
| `CONTENT_ROLE_USER` | `source: "user"` | |
| `CONTENT_ROLE_ASSISTANT` | `source: "agent"` | Only source allowed `metrics`, `model_name`, `tool_calls` |
| `CONTENT_ROLE_SYSTEM` | `source: "system"` | |
| `CONTENT_ROLE_DEVELOPER` | `source: "system"` + `extra.kbench_role` | ATIF has three sources; kbench has more roles |
| `CONTENT_ROLE_CONTEXT` | `source: "system"` + `extra.kbench_role` | Unless it is a tool result — see [§4.2](#42-tool-loops) |
| an unrecognised role | `source: "system"` + a warning | |
| `Part.text` | `step.message` | |
| `senderName` | `step.extra.sender_name` | No ATIF field |
| `Request.metrics.inputTokens` | `metrics.prompt_tokens` | |
| `Request.metrics.outputTokens` | `metrics.completion_tokens` | |
| cost nanodollars, in and out | `metrics.cost_usd` | Summed, `/1e9`. Half a cost is no cost |
| `totalBackendLatencyMs` | `metrics.extra.total_backend_latency_ms` | No ATIF field, but the only per-turn timing kbench has |
| `Request.reasoningTraces` | `reasoning_content` | On the closing reply, like `metrics`. Absent for streamed replies |

Steps are numbered from 1 with no gaps. Only the first carries a `timestamp`, because kbench times the run and not each message.

A plain two-prompt run:

```json
"steps": [
  {"step_id": 1, "source": "user", "message": "Say hello in one word.",
   "extra": {"sender_name": "User"},
   "timestamp": "2026-09-03T00:15:47.648185Z"},
  {"step_id": 2, "source": "agent", "message": "Hi",
   "model_name": "google/gemini-2.5-flash-lite",
   "extra": {"sender_name": "google/gemini-2.5-flash-lite"},
   "metrics": {"prompt_tokens": 7, "completion_tokens": 1, "cost_usd": 1.1e-06,
               "extra": {"total_backend_latency_ms": 168}}}
]
```

### Scores

`results[]` flattens into a flat name-to-number map, one entry per score.

| run.json result | rewards |
|-----------------|---------|
| `numericResult: {value: 0.9}` | `{"score": 0.9}` |
| `numericResult: {value, confidenceInterval}` | `{"score": 0.9, "score_confidence_interval": 0.05}` |
| `numericResult: {}` | `{"score": 0.0}` — proto3 dropped the zero |
| `booleanResult: true` | `{"score": 1.0}` |
| `dictResult: {"is_correct": true, "n": 3}` | `{"is_correct": 1.0, "n": 3.0}` |
| a `PUBLIC` / `PRIVATE` entry | the same, prefixed `public_` / `private_` |

Which entry becomes the **headline** matters, because harbor's midtier displays the first reward and a dict has no reliable order. So the one entry typed `AGGREGATED` — or the only entry, if there is just one — keeps its keys unprefixed. Everything else is prefixed by its type. A run with only a public and a private split therefore has no unprefixed `score` at all, which is correct: neither of them is the overall figure.

An entry that is neither the headline nor a known split gets no prefix and a `result_entry_unprefixed` warning. Two of those overwrite each other.

Only scalar leaves survive. A string, null, list or nested dict is skipped with a `reward_leaf_unusable` warning:

```json
// dictResult in
{"score": 0.75, "is_correct": true, "n": 3.0,
 "gold": "Paris", "nulled": null, "listy": [1, 2], "nested": {"a": 1}}

// rewards out — four warnings recorded
{"score": 0.75, "is_correct": 1.0, "n": 3.0}
```

If no leaf survives, `verifier_result` is omitted entirely. An empty `rewards` would claim a verifier ran and scored nothing, which is a different thing from not scoring.

The rewards appear twice on purpose: authoritatively in `result.json`, and in the trajectory's `final_metrics.extra.kbench_result`, since ATIF itself has no reward field.

---

## 4. Shapes That Map Unobviously

### 4.1 The delegation step

Whenever a run has other trajectories — side chats or dataset rows — the converter appends one extra step listing them:

```json
{"step_id": 3, "source": "system",
 "message": "[delegated] Other trajectories from this run: judge (google/gemini-2.5-flash-lite).",
 "observation": {"results": [{"subagent_trajectory_ref": [
   {"trajectory_id": "Judged-Run #1::Response assessment with google/gemini-2.5-flash-lite-0ea9b787",
    "extra": {"kbench_chat": "judge"}}
 ]}]}}
```

Refs come in two shapes, and which one tells you where to look:

- **`trajectory_id`** — the child is embedded in this same file, under `subagent_trajectories`.
- **`trajectory_path`** — the child is a sibling file in the same directory.

It is a step of its own rather than an attachment to the last real turn: run.json never records which turn opened a side chat, and pinned to the wrong one it would read as that turn's doing.

### 4.2 Tool loops

A tool result arrives as a `CONTENT_ROLE_CONTEXT` message whose text is a JSON blob. It is **not** a step. It folds onto the preceding agent step as a `tool_call` plus a matching `observation` entry:

```json
{"step_id": 2, "source": "agent", "message": "",
 "tool_calls": [
   {"tool_call_id": "mp_fc_0_jm7zaf1f6a86", "function_name": "add",
    "arguments": {"a": 2, "b": 3}},
   {"tool_call_id": "mp_fc_1_ooto647hrlz4", "function_name": "fail_tool",
    "arguments": {}}],
 "observation": {"results": [
   {"source_call_id": "mp_fc_0_jm7zaf1f6a86", "content": "5"},
   {"source_call_id": "mp_fc_1_ooto647hrlz4",
    "content": "Error invoking tool 'fail_tool': boom",
    "extra": {"is_error": true}}]}}
```

Things worth knowing:

- The **call itself is never serialized** by kbench. It is reconstructed from the result blob, which echoes back the name, the arguments and the call id.
- A result with no `call_id` gets an invented one, marked `extra.kbench_synthetic_id`. Harbor pairs a result to its call by id, so an invented id beats pairing by position.
- Older run.json files record a tool loop as a **second conversation** that forks with a copy of the parent. Those are merged back into the main transcript by longest shared prefix, so the loop does not read as a second agent. Messages are compared by what they say, since a fork renumbers ids. Newer runs are already flat.
- A loop that runs out of rounds ends without a closing reply. kbench keeps nothing past the last one, so those turns never reach the file — you get a `transcript_truncated_by_serializer` warning.

### 4.3 Chat rooms and private channels

**Each room is one subagent trajectory, not one per participant.** A room is a single conversation in kbench, so its personas are names on messages rather than separate models. A private channel is its own room, and so its own trajectory.

The parent's own transcript is empty whenever the task did nothing but run the room. kbench opens a conversation for the room itself and records no requests on it, so there is nothing to convert — you get the placeholder step and then the delegation step:

```json
"steps": [
  {"step_id": 1, "source": "system", "message": "[placeholder] No conversation recorded.",
   "timestamp": "..."},
  {"step_id": 2, "source": "system",
   "message": "[delegated] Other trajectories from this run: Narrator, Secret Planning.", ...}
],
"subagent_trajectories": [
  {"agent": {"name": "Narrator", "version": "0.6.1"}, "steps": [...]},
  {"agent": {"name": "Secret Planning", "version": "0.6.1"}, "steps": [...]}
]
```

`"Narrator"` above is not a role — it is the default `name` of a `ChatRoom` that was created without one. A room named at construction shows that name instead, on both the trajectory and the room's own posts.

| kbench | ATIF |
|--------|------|
| a room | one embedded subagent trajectory, `agent.name` = the room name |
| a private channel | its own subagent trajectory |
| a participant | `step.extra.sender_name` on their messages |
| a post by the room itself | a `user` step, sender name = the room name |

**A participant's model is not recorded.** The messages a room stores carry only their parts, their role and a sender name, so nothing says who ran which turn. Room subagents therefore have no `agent.model_name` and their steps no `model_name`, even when the participants were on different models — as Alice, Bob and Carol are above. If you need to know, the run.json's task code is the only source.

`total_steps` counts only the parent's real turns, so a pure room run reports `0` while the token totals are non-zero. That is not a bug — the totals do include every embedded subagent.

### 4.4 Judges and other subchats

Any conversation after the first that is not a tool loop becomes an embedded subagent trajectory, with `trajectory_id` of the form `<run id>::<conversation id>`.

A judge is the one chat named after the model that ran it (`Response assessment with google/gemini-2.5-flash-lite`). The converter recognises the prefix, sets `agent.name` to `"judge"`, and pulls the model out of the rest of the name — it is the only place the file records who graded.

**Several judges on several models are fine.** `agent.name` is the fixed string `"judge"` for all of them, and the model that tells them apart lives in `agent.model_name`, which every subagent trajectory carries in its own right. So the delegation step names each one by model:

```
[delegated] Other trajectories from this run: judge (Alpha), judge (Beta).
```

A subagent with no model — a room, whose participants' models are not recorded — is listed by name alone. Note that none of these is the run's own `agent.model_name`: that is the model being *graded*.

### 4.5 Dataset evaluation

**Rows are sibling trials, not subagents.** Each row already runs as its own kbench run with its own run.json, so it gets its own `.atif.json` and `.result.json` pair. The parent references them by path.

```json
{"step_id": 2, "source": "system",
 "message": "[delegated] Other trajectories from this run: 2 rows.",
 "observation": {"results": [{"subagent_trajectory_ref": [
   {"trajectory_path": "row_qa-run_param_id_0_google_gemini-2.5-flash-lite.atif.json",
    "extra": {"kbench_py_run_id": "row_qa-Run #1"}},
   {"trajectory_path": "row_qa-run_param_id_1_google_gemini-2.5-flash-lite.atif.json",
    "extra": {"kbench_py_run_id": "row_qa-Run #2"}}
 ]}]}}
```

What ties the group together is **`TrialResult.source`**, set to the parent's task name on the parent *and* on every row. A row's own run.json does not record which eval it belonged to, so the value is passed in by the caller — neither side reads its own task name, or they could disagree.

Rows are paired to files **by `pyRunId`, never by position**: under `n_jobs > 1` subruns are appended in completion order, so the array and the files do not line up. A row whose file cannot be found is left out of the refs with a `subrun_path_unknown` warning.

Some situations to expect:

| Situation | What you get |
|-----------|--------------|
| **Multiple models** | One row per (row, model) pair, each a separate file, all under the same `source`. `extra.kbench.subruns[].model_name` says which model each was. |
| **A row crashed** | Still a sibling file, with its own `exception_info`. The parent's `extra.kbench.subruns[]` records its `state` and `error_message`. |
| **Row files deleted** | `remove_run_files=True` removes the pair too. The parent then warns per missing row and keeps only the `extra.kbench.subruns` summary. |
| **A cached row** | Its files are already on disk from the earlier run and are reused as-is. |
| **Cost** | The parent's totals are empty. Each row bills itself. |

### 4.6 Merged aggregates

`merge_results_from_runfiles` builds a parent that has **no conversations at all**, only stubs of the runs it merged. Its placeholder says so:

```
[placeholder] Aggregated from 2 runs. No conversation recorded.
```

Compare with a run that had a conversation and simply never spoke:

```
[placeholder] No conversation recorded.
```

That distinction — aggregate versus silent — is the only thing separating the two, so it is spelled out in the message.

### 4.7 Media ATIF cannot hold

ATIF refers to media **by path**. Inline bytes have nowhere to go. And at v1.7 a content part's type may only be `text` or `image` — anything else is rejected outright, taking the whole file with it.

So exactly one case survives as real content: an **image at a URL**, with a mime type of jpeg, png, gif or webp.

```json
// image at a url — a real content part
"message": [{"type": "image",
             "source": {"media_type": "image/png",
                        "path": "https://www.kaggle.com/static/images/site-logo.png"}}]
```

Everything else becomes a text marker plus a warning:

```json
// inline image — the bytes are kept in extra, the message says what was there
"message": "[image/png: inline data -- not representable in ATIF]",
"extra": {"kbench_media": [{"kind": "inline_data", "mime_type": "image/png",
                            "data": "iVBORw0KGgoAAAANSUhEUg..."}]}
```

| Input | Result |
|-------|--------|
| image at a url, known mime type | a real `image` content part |
| image inline | text marker — no path to point at |
| audio, any form | text marker — v1.8 adds the type, we emit v1.7 |
| video, any form | text marker — ATIF has no video type at any version |

The payload always survives in `step.extra.kbench_media`, whether or not it was representable, so the trajectory alone still says what was sent.

Note that `message` becomes a **list** only when a real media part is present. Every other producer writes a string, so a list is a reliable signal.

### 4.8 Failures

`exception_info` lives in `result.json`. It is keyed off `errorMessage`, not off `state`: an errored run can still carry a result and an intact transcript.

kbench stores a whole `traceback.format_exc()`, so it is split — the last block's exception line gives the type and message, and the raw text becomes the traceback. A chained traceback repeats the header, and the block that ended the run is the last one.

```json
"exception_info": {
  "exception_type": "ValueError",
  "exception_message": "kaboom before any prompt",
  "exception_traceback": "Traceback (most recent call last):\n  File ...",
  "occurred_at": "2026-09-03T00:16:58.536305Z"
}
```

Three details:

- The type is reduced to its bare name, which is the form harbor uses for its own exceptions: `pkg.mod.MyError` becomes `MyError`.
- If the text does not parse as a traceback, the type falls back to `KbenchRunError`, the message is kept verbatim, and `exception_traceback` is absent.
- `occurred_at` is required, so it falls back through `endTime`, `startTime`, then the epoch. A fixed epoch beats `now()`, which would differ every time the same run was converted.

Where the failure shows up in the transcript depends on when it happened:

| When | Transcript |
|------|------------|
| **Before any model turn** | One placeholder step: `[placeholder] Run failed before any model turn: <traceback>` |
| **After a model turn** | The real steps, then `[error] Run failed after the last model turn: ValueError: ...` |

ATIF has no field for a failure, which is why it is appended as a step — otherwise the transcript would just end mid-air.

One rename: harbor reads `AgentTimeoutError` and `VerifierTimeoutError` as completed-without-reward, but a kbench timeout has *failed*. Both are prefixed to `KbenchAgentTimeoutError` and `KbenchVerifierTimeoutError` so they are not misread.

---

## 5. Reading a Trajectory: Symptom Lookup

You opened an `.atif.json` and something looks off. Start here.

| What you see | What it means |
|--------------|---------------|
| `"total_steps": 0` but tokens are non-zero | The parent had no turns of its own. The work happened in a room or a subchat — see [§4.3](#43-chat-rooms-and-private-channels). |
| `total_cost_usd` missing, tokens present | At least one message was unpriced. A partial sum would hide that. |
| A `[placeholder]` step | kbench wrote it, not the model. The wording says which case — [§4.6](#46-merged-aggregates), [§4.8](#48-failures). |
| A `[delegated]` step | There are other trajectories. `trajectory_id` = embedded here, `trajectory_path` = sibling file. |
| A `[error]` step at the end | The run failed after its last model turn. |
| `"arguments": {}` | The tool genuinely took no arguments. Every other empty value would have been dropped. |
| `tool_call_id` starting `kbench_call_` | Invented. The backend supplied no id — look for `extra.kbench_synthetic_id`. |
| `message` is a list, not a string | The message holds a real media part. |
| `[... -- not representable in ATIF]` | Media ATIF cannot reference. The payload is in `extra.kbench_media`. |
| Several agent steps in a row, no user step between | A tool loop. Each round is one agent step carrying that round's calls. See [§4.2](#42-tool-loops). |
| `agent.name` is not `kaggle-benchmarks` | You are inside a subagent trajectory — a room, a channel, or a judge. |
| No `model_name` anywhere in a subagent | A room. Participants' models are not recorded — see [§4.3](#43-chat-rooms-and-private-channels). |
| A `system` step whose text is a JSON tool blob | A tool result with no agent step before it to fold onto. Look for `tool_result_without_call`. |
| `"arguments": {}` on a call that clearly took some | kbench refused the call and quoted them back as text. They are in the result — look for `tool_arguments_unparsed`. |
| No `reasoning_content` on a thinking model | The reply was streamed, or the provider returned none. |
| `extra.kbench.conversion_warnings` | Something was skipped. `kind` says what. |
| A score you expected is missing | Its leaf was not a number. Check the warnings for `reward_leaf_unusable`. |
| No `.atif.json` at all | `WRITE_ATIF=False`, or conversion failed — check the logs for `Could not write`. |

---

## 6. Known Gaps

- **Streamed replies carry no reasoning.** `ModelRequest.reasoning_traces` is only filled when the provider hands it back on a complete response; streaming does not capture it yet, so those steps have no `reasoning_content`.
- **A tool call with no result is lost.** The call is only recoverable from its result blob, so a loop that ends without one drops both.
- **Latency is per turn only.** kbench records no end time per message, so steps after the first carry no `timestamp`.
- **Inline media is described, not written out.** A producer may save the bytes beside the trajectory and point `path` at them — harbor's own antigravity adapter writes an `images/` directory next to the file — which would turn most of [§4.7](#47-media-atif-cannot-hold) into real content parts. Not done here: it makes conversion a writer of arbitrarily many files, and the bytes are still in `extra.kbench_media` either way.
