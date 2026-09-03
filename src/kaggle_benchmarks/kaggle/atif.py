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

"""Converts a run.json into the trajectory and result files harbor tooling reads.

Both converters read the serialized dict rather than live objects, so the C#
backfill can mirror them and their output can be compared. An unmappable field
is skipped, logged and recorded in the output rather than raising.

Fields are written even when empty and dropped by _prune on the way out, so a
mapping reads as one expression rather than as a chain of ifs.
"""

import importlib.metadata
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

Json = dict[str, Any]
Warnings = list[dict[str, str]]

# The newest version a released harbor accepts. v1.8 adds audio content parts
# and nothing else we use, so staying here costs only the audio case below.
SCHEMA_VERSION = "ATIF-v1.7"

# Mark the steps kbench produced and the model did not, so a reader does not
# take one for something that was said. `notes` explains the placeholder at
# length; these are what survive being read on their own, among real steps.
PLACEHOLDER = "[placeholder]"
ERROR = "[error]"
DELEGATED = "[delegated]"

# Both files sit beside the run.json and are named off it. Neither may end in
# "run.json": that suffix is how the backend picks run files out of a directory.
RUN_SUFFIX = ".run.json"
ATIF_SUFFIX = ".atif.json"
TRIAL_SUFFIX = ".result.json"

ASSISTANT_ROLE = "CONTENT_ROLE_ASSISTANT"
DEVELOPER_ROLE = "CONTENT_ROLE_DEVELOPER"
TOOL_ROLE = "CONTENT_ROLE_CONTEXT"
ROLE_TO_SOURCE = {
    "CONTENT_ROLE_USER": "user",
    ASSISTANT_ROLE: "agent",
    "CONTENT_ROLE_SYSTEM": "system",
    DEVELOPER_ROLE: "system",
    TOOL_ROLE: "system",
}

# Where _contents_of parks a request's metrics and reasoning. Not proto fields.
METRICS_KEY = "_requestMetrics"
REASONING_KEY = "_requestReasoning"

# (proto, ATIF). Harbor spells the counts differently from the proto.
TOKEN_FIELDS = (("inputTokens", "prompt_tokens"), ("outputTokens", "completion_tokens"))
COST_FIELDS = ("inputTokensCostNanodollars", "outputTokensCostNanodollars")

# Media ATIF can point at. Images only until we emit v1.8: a content part typed
# anything else is rejected outright, so audio takes the placeholder path with
# everything else we cannot show.
MEDIA_TYPES = frozenset("image/jpeg image/png image/gif image/webp".split())

# Named after the model that graded, the only place the file says who it was.
JUDGE_CHAT_PREFIX = "Response assessment with "
TRACEBACK_HEADER = "Traceback (most recent call last):"
# Harbor reads these as completed-without-reward; a kbench timeout has failed.
HARBOR_TIMEOUTS = ("AgentTimeoutError", "VerifierTimeoutError")

# Represented by the trajectory itself. The rest is copied into extra.kbench,
# so a field a newer kbench adds is carried across regardless.
MAPPED_FIELDS = frozenset(
    {"conversations", "results", "modelVersion", "pyRunId", "id", "startTime", "subruns"}
)  # fmt: skip

# Implausible on purpose: path-shaped invites treating a run as a harbor trial.
TASK_PATH = "kbench://not-a-harbor-task"
TRIAL_URI = "kbench://not-a-harbor-trial"


class ConversionError(Exception):
    """Raised when a run.json cannot produce a valid output file at all."""


def _warn(warnings: Warnings, kind: str, detail: str) -> None:
    # In the file as well as the log: a log line is gone once the notebook closes.
    logger.warning(f"Converting to atif, {kind}: {detail}")
    warnings.append({"kind": kind, "detail": detail})


def _snake(name: str) -> str:
    return "".join(f"_{c.lower()}" if c.isupper() else c for c in name)


def _snake_keys(value: Any) -> Any:
    """Undoes the camelCasing MessageToJson applies on the way out.

    Exactly reverses it, since protobuf capitalises only the letter after each
    underscore: `version_number` can round-trip, `versionNUMBER` cannot occur.
    So this restores the names the .proto declares rather than inventing any,
    and it holds for fields added after this was written.
    """
    if isinstance(value, dict):
        return {_snake(k): _snake_keys(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_snake_keys(v) for v in value]
    return value


def _harness_version() -> str:
    try:
        return importlib.metadata.version("kaggle-benchmarks")
    except importlib.metadata.PackageNotFoundError:
        from kaggle_benchmarks import __version__  # Editable install.

        return __version__


def _int(value: Any) -> int | None:
    """A proto int64, which serializes as a string, not a number."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _prune(node: Any) -> Any:
    """The same structure without None or empty containers."""
    if isinstance(node, dict):
        pruned = ((key, _prune(value)) for key, value in node.items())
        # Empty reads as measured-and-nothing, bar a call with no arguments.
        return {k: v for k, v in pruned if v not in (None, {}, []) or k == "arguments"}
    if isinstance(node, list):
        return [_prune(value) for value in node]
    return node


def _chat_name(conversation_id: str) -> str:
    """The chat's name, without the uuid suffix kbench appends to it."""
    name, _, suffix = conversation_id.rpartition("-")
    return name if name and len(suffix) == 8 else conversation_id


def _media_part(part: Json) -> Json | None:
    if file_data := part.get("fileData"):
        mime, where = file_data.get("mimeType"), file_data.get("fileUri")
        return {"kind": "file_data", "mime_type": mime or "", "uri": where or ""}
    if inline_data := part.get("inlineData"):
        mime, what = inline_data.get("mimeType"), inline_data.get("data")
        return {"kind": "inline_data", "mime_type": mime or "", "data": what or ""}
    return None


def _message_of(content: Json, warnings: Warnings) -> str | list[Json]:
    """A message as plain text, or as content parts when it holds media."""
    parts: list[Json] = []
    for part in content.get("parts") or []:
        media = _media_part(part)
        if media is None:
            parts.append({"type": "text", "text": part.get("text") or ""})
            continue

        mime, uri = media["mime_type"], media.get("uri")
        # ATIF refers to media by path, so inline bytes have nowhere to go.
        if uri and mime in MEDIA_TYPES:
            source = {"media_type": mime, "path": uri}
            parts.append({"type": mime.split("/")[0], "source": source})
            continue
        described = f"{mime or 'unknown media type'}: {uri or 'inline data'}"
        _warn(warnings, "media_unrepresentable", described)
        parts.append(
            {"type": "text", "text": f"[{described} -- not representable in ATIF]"}
        )

    # Harbor allows both, and every other producer writes the plain string.
    if all(part["type"] == "text" for part in parts):
        return "".join(part["text"] for part in parts)
    return parts


def _usage_of(content: Json) -> Json:
    """One message's cost and tokens, under the names ATIF gives them."""
    metrics = content.get(METRICS_KEY) or {}
    usage = {
        key: value
        for proto_field, key in TOKEN_FIELDS
        if (value := _int(metrics.get(proto_field))) is not None
    }
    halves = [_int(metrics.get(field)) for field in COST_FIELDS]
    if any(half is not None for half in halves):
        # Half a cost is an unknown total, not a partial one; None says so.
        usage["cost_usd"] = None if None in halves else sum(halves) / 1_000_000_000
    # Harbor has no latency field, but it is the only per-turn timing there is.
    if (latency := _int(metrics.get("totalBackendLatencyMs"))) is not None:
        usage["extra"] = {"total_backend_latency_ms": latency}
    return usage


def _contents_of(conversation: Json) -> list[Json]:
    """A conversation flattened into messages, cost riding on the closing reply."""
    contents = []
    for request in conversation.get("requests") or []:
        group = list(request.get("contents") or [])
        if group and group[-1].get("role") == ASSISTANT_ROLE:
            # Both belong to the reply that closed the request, not the request.
            rider = {
                METRICS_KEY: request.get("metrics"),
                REASONING_KEY: request.get("reasoningTraces"),
            }
            group[-1] = {**group[-1], **{k: v for k, v in rider.items() if v}}
        contents.extend(group)
    return contents


def _partition(
    conversations: list[Json], warnings: Warnings
) -> tuple[list[Json], list[Json]]:
    """One main transcript and any side chats."""

    # A tool loop forks with a copy of the parent, and is merged back so it
    # does not read as a second agent.

    def shared_prefix(main: list[Json], contents: list[Json]) -> int:
        # By what a message says, since a fork renumbers ids.
        def key(content: Json) -> str:
            said = [content.get(f) for f in ("role", "senderName", "parts")]
            return json.dumps(said, sort_keys=True)

        pairs = enumerate(zip(map(key, main), map(key, contents)))
        unshared = (n for n, (left, right) in pairs if left != right)
        return next(unshared, min(len(main), len(contents)))

    if not conversations:
        return [], []
    main = _contents_of(conversations[0])
    side_chats = []
    for conversation in conversations[1:]:
        contents = _contents_of(conversation)
        shared = shared_prefix(main, contents)
        is_loop = _chat_name(conversation["id"]) == "Tool loop"
        if is_loop and (shared or not main):
            # Spliced where it branched, which need not be the parent's end.
            main = main[:shared] + contents[shared:] + main[shared:]
            continue
        if is_loop:
            # Merging would interleave two unrelated threads.
            _warn(warnings, "fork_without_common_prefix", conversation["id"])
        side_chats.append(conversation)
    return main, side_chats


def _steps(
    contents: list[Json], model: str | None, started_at: str | None, warnings: Warnings
) -> list[Json]:
    """A transcript as steps, numbered from 1 with no gaps."""

    def tool_blob(content: Json) -> Json | None:
        """A tool result's payload, or None: it shares its role with context."""
        if content.get("role") != TOOL_ROLE:
            return None
        try:
            text = "".join(p.get("text") or "" for p in content.get("parts") or [])
            blob = json.loads(text)
            if isinstance(blob, str):
                blob = json.loads(blob)  # Encoded twice on the way in.
        except ValueError:
            return None
        # Every tool result has both keys; plain context has neither.
        is_result = isinstance(blob, dict) and {"name", "arguments"} <= blob.keys()
        return blob if is_result else None

    def fold(step: Json, blob: Json) -> None:
        """Records one call and its outcome on the agent step that made it."""
        calls = step.setdefault("tool_calls", [])
        results = step.setdefault("observation", {"results": []})["results"]
        args, given = blob.get("arguments"), blob.get("call_id")
        if not isinstance(args, (dict, type(None))):
            # kbench refused the call and quoted the arguments back, so they
            # survive in the result below rather than in arguments.
            _warn(warnings, "tool_arguments_unparsed", f"{blob.get('name')}: {args!r}")
        # Harbor links result to call by id; position is all we have without.
        call_id = given or f"kbench_call_{step['step_id']}_{len(calls) + 1}"
        calls.append(
            {
                "tool_call_id": call_id,
                "function_name": blob.get("name") or "",
                "arguments": args if isinstance(args, dict) else {},
                "extra": None if given else {"kbench_synthetic_id": True},
            }
        )
        # Harbor wants text, and a tool can return any value.
        error, output = blob.get("error"), blob.get("output")
        said = output if isinstance(output, str) else json.dumps(output)
        results.append(
            {
                "source_call_id": call_id,
                "content": error or ("" if output is None else said),
                "extra": {"is_error": True} if error else None,
            }
        )

    steps: list[Json] = []
    for content in contents:
        blob = tool_blob(content)
        # A tool result is the outcome of a call, and only an agent step has one.
        if blob and steps and steps[-1]["source"] == "agent":
            fold(steps[-1], blob)
            continue
        if blob:
            _warn(warnings, "tool_result_without_call", str(blob.get("name")))

        role = content.get("role")
        if role not in ROLE_TO_SOURCE:
            _warn(warnings, "unknown_content_role", f"{role!r} treated as system")
        source = ROLE_TO_SOURCE.get(role, "system")
        agent = source == "agent"
        kbench_role = role if role in (DEVELOPER_ROLE, TOOL_ROLE) else None
        media = [m for m in map(_media_part, content.get("parts") or []) if m]
        steps.append(
            {
                "step_id": len(steps) + 1,
                "source": source,
                "message": _message_of(content, warnings),
                "extra": {
                    "sender_name": content.get("senderName"),
                    # Both land on system; the role keeps them tellable apart.
                    "kbench_role": kbench_role,
                    # Shown or not, so the trajectory alone says what was sent.
                    "kbench_media": media,
                },
                # Harbor rejects the trajectory if a non-agent step carries these.
                "model_name": model if agent else None,
                "metrics": _usage_of(content) if agent else None,
                "reasoning_content": content.get(REASONING_KEY) if agent else None,
            }
        )

    # kbench times the run, not each message, and all alike reads as instant.
    if steps and started_at:
        steps[0]["timestamp"] = started_at
    return steps


def _subagent(
    conversation: Json, run_id: str, started_at: str | None, warnings: Warnings
) -> Json:
    """One side chat as a subagent trajectory, complete in itself."""
    # One subagent per room, not per persona: a room is one conversation, so
    # its personas are names on messages, not models.
    name = _chat_name(conversation["id"])
    # Only a judge reveals which model it ran on.
    is_judge = name.startswith(JUDGE_CHAT_PREFIX)
    model = name[len(JUDGE_CHAT_PREFIX) :] or None if is_judge else None
    steps = _steps(_contents_of(conversation), model, started_at, warnings)
    agent = {"name": "judge" if is_judge else name, "model_name": model}
    return {
        "schema_version": SCHEMA_VERSION,
        "trajectory_id": f"{run_id}::{conversation['id']}",
        "agent": {**agent, "version": _harness_version()},
        # Harbor wants a step, and a room can be posted in and never answered.
        "steps": steps
        or [
            {
                "step_id": 1,
                "source": "system",
                "message": f"{PLACEHOLDER} No conversation recorded.",
            }
        ],
    }


def _rewards(run_json: Json, warnings: Warnings) -> dict[str, float] | None:
    """A run's scores as the name-to-number map harbor reads."""

    def flatten(result: Json, prefix: str) -> dict[str, float]:
        if "numericResult" in result:
            # proto3 omits a zero-valued scalar, so an absent value is 0.0.
            numeric = result["numericResult"] or {}
            leaves = {"score": numeric.get("value", 0.0)}
            if (interval := numeric.get("confidenceInterval")) is not None:
                leaves["score_confidence_interval"] = interval
        elif "booleanResult" in result:
            leaves = {"score": result["booleanResult"]}
        else:
            leaves = result.get("dictResult") or {}

        rewards = {}
        for key, value in leaves.items():
            # bool is an int subclass, so a pass/fail flag lands here as 1.0/0.0.
            if isinstance(value, (int, float)):
                rewards[f"{prefix}{key}"] = float(value)
            else:
                kind = type(value).__name__
                _warn(warnings, "reward_leaf_unusable", f"{prefix}{key} is a {kind}")
        return rewards

    # The midtier shows rewards.First(), so the overall figure takes the
    # unprefixed key however the splits were ordered.
    results = list(run_json.get("results") or [])
    aggregate = next((r for r in results if r.get("type") == "AGGREGATED"), None)
    headline = aggregate or (results[0] if len(results) == 1 else None)

    rewards = flatten(headline, "") if headline else {}
    for result in (r for r in results if r is not headline):
        kind = result.get("type", "")
        prefix = {"PUBLIC": "public_", "PRIVATE": "private_"}.get(kind, "")
        if not prefix:
            _warn(warnings, "result_entry_unprefixed", f"result type {kind!r}")
        rewards.update(flatten(result, prefix))
    return rewards or None


def _exception_info(run_json: Json) -> Json | None:
    """How the run failed, or None if it did not."""
    # Keyed off the message, not the state: an errored run can still carry a
    # result and an intact transcript.
    if not (error := run_json.get("errorMessage")):
        return None

    lines = error.rstrip().splitlines()
    # A chained traceback repeats the header and its last block ended the run.
    # Frames are indented and the exception line is not, so it starts at the
    # first unindented line after the last header.
    headers = [i for i, line in enumerate(lines) if line == TRACEBACK_HEADER]
    rest = range(headers[-1] + 1, len(lines)) if headers else range(0)
    # 0 for not found, which no real exception line can be: a header precedes it.
    start = next((i for i in rest if lines[i][:1].strip()), 0)
    # Split once: the message may hold a colon and may span lines. The proto
    # field is free text, so a non-traceback degrades rather than guessing.
    name, _, said = "\n".join(lines[start:]).partition(": ") if start else ("", "", "")
    # Bare name, the form harbor uses for its own exceptions.
    exception_type = name.rpartition(".")[2] or name or "KbenchRunError"
    if exception_type in HARBOR_TIMEOUTS:
        exception_type = f"Kbench{exception_type}"
    return {
        "exception_type": exception_type,
        "exception_message": said if start else error,
        "exception_traceback": error if start else "",
        # Required, and a fixed epoch beats now(), which would differ per run.
        "occurred_at": run_json.get("endTime")
        or run_json.get("startTime")
        or "1970-01-01T00:00:00Z",
    }


def _task_name(run_json: Json) -> str:
    # Harbor requires it, and there is nothing to fall back to.
    if not (name := (run_json.get("taskVersion") or {}).get("name")):
        raise ConversionError("run.json has no taskVersion.name")
    return str(name)


def _run_id(run_json: Json) -> str:
    return str(run_json.get("pyRunId") or run_json.get("id") or "unknown")


def _model_slug(run_json: Json) -> str | None:
    return (run_json.get("modelVersion") or {}).get("slug") or None


def _kbench_extra(run_json: Json, warnings: Warnings) -> Json:
    """Everything with no ATIF field, so a reader can get back to the run.json."""
    kbench = {
        _snake(f): _snake_keys(v) for f, v in run_json.items() if f not in MAPPED_FIELDS
    }
    # How each row turned out, minus its transcript. Own model: evals compare.
    kbench["subruns"] = [
        {
            "py_run_id": subrun.get("pyRunId"),
            "task_name": (subrun.get("taskVersion") or {}).get("name"),
            "model_name": _model_slug(subrun),
            "state": subrun.get("state"),
            "rewards": _rewards(subrun, warnings),
            "error_message": subrun.get("errorMessage"),
        }
        for subrun in run_json.get("subruns") or []
    ]
    # Last, so it holds every warning the conversion raised.
    kbench["conversion_warnings"] = warnings
    return kbench


def to_atif(run_json: Json, subrun_paths: dict[str, str] | None = None) -> Json:
    """A run.json as an ATIF trajectory.

    `subrun_paths` maps a dataset eval's row ids to their trajectory files; only
    the caller knows them. Raises ConversionError if the run has no task name.
    """
    task_name = _task_name(run_json)
    run_id = _run_id(run_json)
    model = _model_slug(run_json)
    started_at = run_json.get("startTime")
    subruns = run_json.get("subruns") or []
    warnings: Warnings = []
    main, side_chats = _partition(run_json.get("conversations") or [], warnings)
    steps = _steps(main, model, started_at, warnings)
    subagents = [_subagent(chat, run_id, started_at, warnings) for chat in side_chats]
    if not steps:
        # Harbor wants a step, and which absence it was matters: only an
        # aggregate has no conversation at all, while a run that merely said
        # nothing still has an empty one of its own.
        error = run_json.get("errorMessage")
        if not run_json.get("conversations"):
            said = f"Aggregated from {len(subruns) or 'other'} runs."
        else:
            said = f"Run failed before any model turn: {error}" if error else ""
        message = " ".join(
            part for part in (PLACEHOLDER, said, "No conversation recorded.") if part
        )
        step = {"step_id": 1, "source": "system", "message": message}
        steps = [{**step, "timestamp": started_at}]

    # Zero when the only step is a placeholder; notes explains the disagreement.
    # Before the steps kbench adds below, which are not turns it counts.
    real_steps = len(steps) if main else 0

    # A side chat rides inside this file; an eval's rows are sibling files the
    # parent does not name. By id, never position: rows finish out of order.
    refs = [
        {"trajectory_id": sub["trajectory_id"], "extra": {"kbench_chat": chat}}
        for sub in subagents
        if (chat := sub["agent"]["name"])
    ]
    named = [sub["agent"]["name"] for sub in subagents if sub["agent"]["name"]]
    for row_id in [subrun.get("pyRunId") for subrun in subruns]:
        if not (path := (subrun_paths or {}).get(row_id)):
            _warn(warnings, "subrun_path_unknown", f"{row_id}")
            continue
        refs.append({"trajectory_path": path, "extra": {"kbench_py_run_id": row_id}})
    if subruns:
        named.append(f"{len(subruns)} rows")
    if refs:
        # A step of its own, rather than hung off the last real turn: run.json
        # never records which turn opened a side chat, so any turn we chose
        # would be a guess, and on the wrong one the chat reads as that turn's
        # doing. This one says only that they ran, which is all the file knows.
        steps.append(
            {
                "step_id": len(steps) + 1,
                "source": "system",
                "message": f"{DELEGATED} Other trajectories from this run: "
                f"{', '.join(named)}.",
                "observation": {"results": [{"subagent_trajectory_ref": refs}]},
            }
        )

    failure = _exception_info(run_json) or {}
    if failure.get("exception_type") == "ToolInvocationLimitExhausted":
        # A loop out of rounds sends no closing reply, and kbench keeps nothing
        # past the last one: the turns are gone before we see the file.
        _warn(warnings, "transcript_truncated_by_serializer", failure["exception_type"])

    if failure and main:
        # The transcript is the one file with nowhere to say a run ended badly:
        # ATIF has no field for it, and exception_info lives in the other file.
        # Only where there are turns to end -- a run that failed before its
        # first says so in the placeholder above, and would say it twice here.
        # Last of all, since it is how the run stopped.
        steps.append(
            {
                "step_id": len(steps) + 1,
                "source": "system",
                "timestamp": failure["occurred_at"],
                "message": f"{ERROR} Run failed after the last model turn: "
                f"{failure['exception_type']}: {failure['exception_message']}",
            }
        )

    # Off the steps, so a total cannot disagree with the transcript and a
    # fork's copy of its parent's costs is counted once. Absent, not zero:
    # unmeasured is not free.
    told = steps + [step for sub in subagents for step in sub["steps"]]
    spent = [step["metrics"] for step in told if step.get("metrics")]
    totals = {
        key: sum(usage[key] for usage in spent if key in usage)
        for _, key in TOKEN_FIELDS
        if any(key in usage for usage in spent)
    }
    # One unpriced message makes the bill unknowable; a partial sum hides that.
    costs = [usage["cost_usd"] for usage in spent if "cost_usd" in usage]

    note = f"Converted from kbench run.json for task {task_name!r}."
    if not real_steps:
        note += (
            " No model turn was recorded, so total_steps is 0 and the first"
            " step below is a placeholder."
        )
    return _prune(
        {
            "schema_version": SCHEMA_VERSION,
            "session_id": run_id,
            "trajectory_id": run_id,
            "agent": {
                "name": "kaggle-benchmarks",
                "version": _harness_version(),
                "model_name": model,
            },
            "steps": steps,
            "notes": note,
            "final_metrics": {
                "total_prompt_tokens": totals.get("prompt_tokens"),
                "total_completion_tokens": totals.get("completion_tokens"),
                "total_cost_usd": None if not costs or None in costs else sum(costs),
                "total_steps": real_steps,
                # ATIF has no reward field, so it rides in ours instead.
                "extra": {"kbench_result": _rewards(run_json, warnings)},
            },
            "extra": {"kbench": _kbench_extra(run_json, warnings)},
            "subagent_trajectories": subagents,
        }
    )


def to_trial_result(
    run_json: Json, source: str | None = None, trajectory: Json | None = None
) -> Json:
    """A run.json as harbor's TrialResult.

    `source` groups trials into a dataset, so an eval passes its parent task
    name on the parent and every row alike; a row's own file does not record
    which eval it belonged to. Raises ConversionError if there is no task name.

    Pass `trajectory` when one has already been converted: converting a second
    one logs every warning twice, and this one would not have the subrun paths.
    """
    # Read back off the trajectory, so the two files cannot come to disagree.
    metrics = (trajectory or to_atif(run_json))["final_metrics"]
    model = _model_slug(run_json)
    return _prune(
        {
            "task_name": _task_name(run_json),
            "source": source,
            "trial_name": _run_id(run_json),
            "trial_uri": TRIAL_URI,
            "task_id": {"path": TASK_PATH},
            "task_checksum": "unknown (kbench)",
            "config": {"task": {"path": TASK_PATH}},
            "agent_info": {
                # The harness. No provider on model_info: none is recorded.
                "name": "kaggle-benchmarks",
                "version": _harness_version(),
                "model_info": {"name": model} if model else None,
            },
            "agent_result": {
                "n_input_tokens": metrics.get("total_prompt_tokens"),
                "n_output_tokens": metrics.get("total_completion_tokens"),
                "cost_usd": metrics.get("total_cost_usd"),
            },
            "verifier_result": {
                "rewards": metrics.get("extra", {}).get("kbench_result")
            },
            "exception_info": _exception_info(run_json),
            "started_at": run_json.get("startTime"),
            "finished_at": run_json.get("endTime"),
            # No id: harbor defaults a uuid4, so the same input reconverts alike.
        }
    )


def paths_beside(run_path: str | Path) -> list[Path]:
    """The two files write_beside would write for this run.json."""
    run_path = Path(run_path)
    stem = run_path.name.removesuffix(RUN_SUFFIX)
    return [run_path.with_name(stem + s) for s in (ATIF_SUFFIX, TRIAL_SUFFIX)]


def remove_beside(run_path: str | Path) -> None:
    """Deletes both files. Missing is normal: conversion is allowed to fail."""
    for path in paths_beside(run_path):
        try:
            path.unlink(missing_ok=True)
        except OSError as e:
            logger.warning(f"Could not remove {path}: {e}")


def write_beside(
    run_json: Json,
    run_path: str | Path,
    subrun_paths: dict[str, str] | None = None,
    source: str | None = None,
) -> None:
    """Writes the trajectory and result files next to an already-written run.json.

    Does nothing when `config.write_atif` is off. Each write is wrapped on its
    own: the two conversions share helpers, so the one that fails is not always
    the one you would guess, and neither is worth losing a finished run over.
    """
    from kaggle_benchmarks._config import config

    if not config.write_atif:
        return

    def write(path: Path, convert) -> Json | None:
        try:
            converted = convert()
            path.write_text(json.dumps(converted, indent=2))
            return converted
        except Exception as e:
            logger.warning(f"Could not write {path}: {e}")
            return None

    atif_path, trial_path = paths_beside(run_path)
    # Converted once and handed on. Converting a second one would log every
    # warning again, and would warn that it has no subrun paths -- which only
    # this caller ever had. None if it failed, and to_trial_result converts its
    # own: a trajectory too broken to write can still hold usable totals.
    trajectory = write(atif_path, lambda: to_atif(run_json, subrun_paths=subrun_paths))
    write(
        trial_path,
        lambda: to_trial_result(run_json, source=source, trajectory=trajectory),
    )
