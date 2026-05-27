# Copyright 2025 Kaggle Inc.
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

import ast
import datetime
import functools
import inspect
import json
import logging
import textwrap
from pathlib import Path
from typing import Any, Callable, Tuple, Union

from google.protobuf import json_format

from kaggle_benchmarks import (
    assertions,
    chats,
    content_types,
    results,
    runs,
    tasks,
    utils,
)
from kaggle_benchmarks import messages as benchmark_messages
from kaggle_benchmarks.kaggle import benchmark_types_pb2 as types

TASK_FILE_SUFFIX = ".task.json"
RUN_FILE_SUFFIX = ".run.json"
TASK_RUN_FILENAME_DELIMITER = "-"
_OPTIONAL_AGGREGATED_METRICS = (
    "input_tokens_cost_nanodollars",
    "output_tokens_cost_nanodollars",
    "total_backend_latency_ms",
)

logger = logging.getLogger(__name__)


def dump_task(task: tasks.Task) -> types.BenchmarkTaskVersion:
    return json_format.ParseDict(
        prepare_task(task),
        types.BenchmarkTaskVersion(),
        ignore_unknown_fields=True,
    )


def dump_run(run: runs.Run) -> types.BenchmarkTaskRun:
    return json_format.ParseDict(
        prepare_run(run),
        types.BenchmarkTaskRun(),
        ignore_unknown_fields=True,
    )


def generate_task_filename(task_name: str) -> str:
    """Generates a consistent filename for a benchmark task definition."""
    normalized_task_name = utils.normalize_name(task_name)
    return f"{normalized_task_name}{TASK_FILE_SUFFIX}"


def generate_run_filename(task_name: str, run_id: str) -> str:
    """Generates a consistent filename based on task name and a run ID."""
    normalized_task_name = utils.normalize_name(task_name)
    normalized_run_id = utils.normalize_name(run_id)
    return f"{normalized_task_name}{TASK_RUN_FILENAME_DELIMITER}{normalized_run_id}{RUN_FILE_SUFFIX}"


def get_runs_filenames(runs: runs.Runs) -> list[str]:
    return [generate_run_filename(run.task.name, run.cache_id) for run in runs]


def remove_runs_files(runs: runs.Runs) -> None:
    for run in runs:
        run_filename = generate_run_filename(run.task.name, run.cache_id)
        try:
            Path(run_filename).unlink()
        except Exception as e:
            logger.warning(f"Failed to remove file {run_filename}: {e}")


def _get_source_code(func) -> str:
    if isinstance(func, functools.partial):
        func = func.func
    try:
        return inspect.getsource(func)
    except (OSError, TypeError):
        logger.warning(f"Could not get source code for {func.__name__}")
        return ""


def _find_subtask_names(task_obj: tasks.Task) -> list[str]:
    """Attempts to find subtask names called within a given task's function."""
    subtask_names = []
    try:
        source_code = textwrap.dedent(inspect.getsource(task_obj.func))
        tree = ast.parse(source_code)

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "run"
            ):
                var_expr = ast.unparse(node.func.value)
                # We restrict to global subtasks (excluding nested).
                if "." not in var_expr and var_expr in task_obj.func.__globals__:
                    potential_task_instance = task_obj.func.__globals__[var_expr]
                    if (
                        isinstance(potential_task_instance, tasks.Task)
                        and potential_task_instance.name not in subtask_names
                    ):
                        subtask_names.append(potential_task_instance.name)
        subtask_names.sort()
    except (TypeError, OSError) as e:
        logger.warning(
            f"Could not get source code for task '{task_obj.name}' to find subtasks: {e}"
        )
    return subtask_names


def prepare_run(run: runs.Run) -> dict[str, Any]:
    """
    Prepares a dictionary representation of a tasks.Run object,
    mirroring the structure of BenchmarkTaskRun proto for JSON serialization.
    """
    # TODO (b/419612137): how to keep this auto-synced with proto.
    model_version = _extract_model_version_data(run)
    conversation_entries, assertion_map = _prepare_conversations_data(
        run.chat,
    )
    return {
        "py_run_id": f"{run.task.name}-{run.id}",
        "manual_user_id": None,
        "task_version": {
            "id": None,
            "task_id": None,
            "version_number": run.task.version,
            "name": run.task.name,  # Raw name
            "description": run.task.description,
            "definition": _get_source_code(run.task.func),
        },
        "model_version": model_version,
        "state": _get_run_state_proto(run.status),
        "start_time": _format_timestamp(run.start_time),
        "end_time": _format_timestamp(run.end_time),
        "conversations": conversation_entries,
        "results": _prepare_results_data(run),
        "assertions": _prepare_assertions_data(run.assertion_results, assertion_map),
        "subruns": [prepare_run(subrun) for subrun in run.subruns],
        "error_message": run.error_message,
    }


def prepare_task(task: tasks.Task) -> dict[str, Any]:
    subtask_names = _find_subtask_names(task)

    return {
        "id": None,  # Not available
        "task_id": None,  # Not available
        "version_number": task.version,
        "name": task.name,
        "description": task.description or None,
        "definition": _get_source_code(task.func),
        "subtask_file_names": [
            generate_task_filename(subtask_name) for subtask_name in subtask_names
        ],
    }


def _format_timestamp(dt: datetime.datetime | None) -> str | None:
    """Formats a datetime object to an ISO 8601 string with Z denoting UTC."""
    if dt:
        # Standard library dt.isoformat() for a UTC datetime typically produces
        # a string ending in "+00:00". This replaces it with "Z" (Zulu)
        # for a common UTC representation.
        # If "+00:00" is not present (e.g., already "Z", non-UTC, or naive),
        # .replace() will not alter the string.
        return dt.isoformat().replace("+00:00", "Z")
    return None


def _extract_model_version_data(run: runs.Run) -> dict[str, Any | None]:
    """Extracts model version information from run parameters."""
    actor_param_name = param.name if (param := run.evaluated_subject) else None
    return {
        "id": None,
        "benchmark_model_id": None,
        "external_url": None,
        "knowledge_cutoff": None,
        "is_default": None,
        "slug": actor_param_name,
    }


def _get_run_state_proto(run_status: utils.Status | str) -> types.BenchmarkTaskRunState:
    """Maps a run status string to its corresponding protobuf enum."""
    RUN_STATUS_MAP = {
        utils.Status.PENDING: types.BenchmarkTaskRunState.BENCHMARK_TASK_RUN_STATE_QUEUED,
        utils.Status.RUNNING: types.BenchmarkTaskRunState.BENCHMARK_TASK_RUN_STATE_RUNNING,
        utils.Status.SUCCESS: types.BenchmarkTaskRunState.BENCHMARK_TASK_RUN_STATE_COMPLETED,
        utils.Status.FAILED: types.BenchmarkTaskRunState.BENCHMARK_TASK_RUN_STATE_ERRORED,
    }
    return RUN_STATUS_MAP.get(
        run_status, types.BenchmarkTaskRunState.BENCHMARK_TASK_RUN_STATE_UNSPECIFIED
    )


def _message_to_proto_content(message: benchmark_messages.Message) -> dict[str, Any]:
    """Converts a benchmark_messages.Message to a protobuf Content dictionary."""
    # TODO (b/419612137): Need a better way to keep these synced.
    ROLE_TO_PROTO_ROLE_MAP = {
        "user": types.ContentRole.CONTENT_ROLE_USER,
        "assistant": types.ContentRole.CONTENT_ROLE_ASSISTANT,
        "system": types.ContentRole.CONTENT_ROLE_SYSTEM,
        "developer": types.ContentRole.CONTENT_ROLE_DEVELOPER,
        "tool": types.ContentRole.CONTENT_ROLE_CONTEXT,
    }
    proto_role = ROLE_TO_PROTO_ROLE_MAP.get(
        message.sender.role, types.ContentRole.CONTENT_ROLE_UNSPECIFIED
    )

    part_value: dict[str, Any] = {}  # Holds the 'image' or 'text' part

    if isinstance(
        message.content,
        (content_types.images.ImageURL, content_types.videos.VideoContent),
    ):
        mime = message.content.to_mime()
        part_value["file_data"] = {
            "file_uri": mime["location"],
            "mime_type": mime["mime_type"],
        }
    elif isinstance(
        message.content,
        (content_types.images.ImageBase64, content_types.audios.AudioContent),
    ):
        mime = message.content.to_mime()
        part_value["inline_data"] = {
            "data": mime["content"],
            "mime_type": mime["mime_type"],
        }
    elif isinstance(message.content, str):  # text payload
        part_value["text"] = message.content
    else:  # payload of dict or more complicated objects
        payload = message.payload
        try:
            json_text = json.dumps(payload)
        except TypeError:
            logger.warning(
                f"Could not serialize message payload to JSON for message from {message.sender.name}. "
                f"Payload type: {type(payload)}. Falling back to string representation."
            )
            json_text = str(payload)
        part_value["text"] = json_text

    return {
        "parts": [part_value],
        "role": proto_role,
        "sender_name": message.sender.name,
    }


def _prepare_conversations_data(
    chat: chats.Chat | None,
    parent_id: str = "",
) -> tuple[list[dict[str, Any]], dict[str, tuple[str, str]]]:
    """
    Converts a benchmark `chats.Chat` object, including any nested sub-chats,
    into a list of conversation entries, and a mapping of assertions
    to their conversational context.

    Each conversation entry in the list corresponds to a `Conversation` protobuf
    message structure, containing requests made during that chat. A request
    is defined as a sequence of messages ending with an assistant's response.
    Assertions encountered are mapped to their parent chat ID and the specific
    request (the previous request) ID they link to, facilitating contextual display.

    Each conversation entry includes a ``parent_conversation_id`` field linking
    it to its parent conversation, enabling consumers to reconstruct the chat
    tree hierarchy.

    Args:
        chat: The `chats.Chat` object to process.
        parent_id: The ID of the parent conversation, used to populate the
            ``parent_conversation_id`` field on child conversations.

    Returns:
        A tuple containing:
            - A list of conversation entries.
            - A dictionary mapping assertion IDs to (chat_id, request_id) tuples.
    """
    if not chat:
        return [], {}

    current_conversation_requests = []
    current_request_contents = []
    subchat_conversation_entries = []
    request_counter = 0

    conversation_assertion_map = {}

    for item in chat.history:
        if isinstance(item, benchmark_messages.Message):
            # Associate assertion with its parent chat and the specific request it links to.
            # This allows UI or reporting tools to display assertions within their relevant conversational context.
            if item.sender.name == "Assertion" and isinstance(
                item.content, assertions.AssertionResult
            ):
                previous_request_id = (
                    current_conversation_requests[-1]["id"]
                    if current_conversation_requests
                    else None
                )
                conversation_assertion_map[item.content.id] = (
                    chat.id,
                    previous_request_id,
                )

            # Ignore messages invisible to llm in requests.
            if not item.is_visible_to_llm:
                continue

            current_request_contents.append(_message_to_proto_content(item))

            # A request is formed when an assistant message is encountered.
            # The sender role is checked on the original message object.
            # Any messages in after the last "assistant" message will be ignored.
            if item.sender.role == "assistant":
                if (
                    current_request_contents
                ):  # Ensure there's something to form a request
                    request_metrics = {
                        "input_tokens": item._meta.get("input_tokens"),
                        "output_tokens": item._meta.get("output_tokens"),
                        "input_tokens_cost_nanodollars": item._meta.get(
                            "input_tokens_cost_nanodollars"
                        ),
                        "output_tokens_cost_nanodollars": item._meta.get(
                            "output_tokens_cost_nanodollars"
                        ),
                        "total_backend_latency_ms": item._meta.get(
                            "total_backend_latency_ms"
                        ),
                    }
                    request_counter += 1
                    current_conversation_requests.append(
                        {
                            "id": f"{chat.id}-req-{request_counter}",
                            "contents": list(current_request_contents),
                            "metrics": request_metrics,
                        }
                    )
                    current_request_contents.clear()

        elif isinstance(item, chats.Chat):
            # Recursively prepare conversation data for the sub-chat
            # and collect its entries to be added after the current chat's entry.
            conversation_entries, assertion_map = _prepare_conversations_data(
                item, parent_id=chat.id
            )

            conversation_assertion_map.update(assertion_map)
            subchat_conversation_entries.extend(conversation_entries)
        else:
            raise NotImplementedError(
                f"Unhandled item type in chat history: {type(item)} for chat '{chat.name}'"
            )

    conversation_metrics = {
        "input_tokens": 0,
        "output_tokens": 0,
        "input_tokens_cost_nanodollars": None,
        "output_tokens_cost_nanodollars": None,
        "total_backend_latency_ms": None,
    }
    for request in current_conversation_requests:
        if metrics := request.get("metrics"):
            conversation_metrics["input_tokens"] += metrics.get("input_tokens") or 0
            conversation_metrics["output_tokens"] += metrics.get("output_tokens") or 0
            for field in _OPTIONAL_AGGREGATED_METRICS:
                if (value := metrics.get(field)) is not None:
                    conversation_metrics[field] = (
                        conversation_metrics[field] or 0
                    ) + value
    current_conversation_entry = {
        "id": chat.id,
        "requests": current_conversation_requests,
        "metrics": conversation_metrics,
        "model_version_slug": "model_version_slug for conversation is DEPRECATED",
        "parent_conversation_id": parent_id or None,
    }

    conversation_entries = [current_conversation_entry] + subchat_conversation_entries

    return conversation_entries, conversation_assertion_map


def _is_tuple_result(value: Any) -> bool:
    """Checks if a value is a tuple of two numbers (int or float)."""
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and all(isinstance(item, (int, float)) for item in value)
    )


def _prepare_results_data(run: runs.Run) -> list[dict[str, Any]]:
    """Helper function to prepare the 'results' list for a BenchmarkTaskRun."""
    result_data: dict[str, Any]
    # We cannot use run.task.result_type to decide here since the run result could be
    # loaded from the cache, where its task.result_type information is lost.

    # Represented as numeric result for both PassCount or MetricWithCI
    if _is_tuple_result(run.result):
        result_data = {
            "numericResult": {
                "value": run.result[0],
                "confidenceInterval": run.result[1],
            }
        }
    elif run.result is None:
        result_data = {"boolean_result": run.passed}
    elif run.result is results.FAILED:
        result_data = {"boolean_result": False}
    elif isinstance(run.result, bool):
        result_data = {"boolean_result": run.result}
    elif isinstance(run.result, (int, float)):
        result_data = {"numeric_result": {"value": float(run.result)}}
    elif isinstance(run.result, dict):
        result_data = {"dict_result": run.result}
    else:  # Any other unhandled Result type
        raise NotImplementedError(
            f"Result type cannot be serialized to show on leaderboard: {type(run.result)}."
        )

    result_data |= {"type": types.BenchmarkTaskRunResultType.AGGREGATED}
    return [result_data]


def _prepare_assertions_data(
    assertion_results: list[assertions.AssertionResult],
    assertion_map: dict[str, tuple[str, str]],
) -> list[dict[str, Any]]:
    """Helper function to prepare the 'assertions' list for a BenchmarkTaskRun."""
    assertions_list_data = []
    for assertion_result in assertion_results:
        details = assertion_result.details or {}
        chat_id, request_id = assertion_map.get(assertion_result.id, (None, None))
        assertions_list_data.append(
            {
                "expectation": assertion_result.expectation,
                "status": types.BenchmarkTaskRunAssertionStatus.BENCHMARK_TASK_RUN_ASSERTION_STATUS_PASSED
                if assertion_result.passed
                else types.BenchmarkTaskRunAssertionStatus.BENCHMARK_TASK_RUN_ASSERTION_STATUS_FAILED,
                "line_number": details.get("line_number"),
                "definition": details.get("source_code"),
                "conversation_request_ids": [
                    {
                        "conversation_id": chat_id,
                        "request_id": request_id,
                    }
                ],
            }
        )
    return assertions_list_data


def merge_results_from_runfiles(
    run_files: list[str],
    aggregate_fn: Callable[[list[Any]], Union[bool, float, Tuple[float, float]]],
    output_run_id: str = "Run aggregated",
    output_directory: str | None = None,
    delete_run_files: bool = False,
) -> str:
    """
    Merges multiple run files into a single run file with an aggregated result.
    This is to merge results only, so conversations and assertions are ignored for now.
    This function reads a list of run JSON files, validates their consistency,
    applies an aggregation function to their results, and writes a new
    "aggregated" run file.
    Args:
        run_files: A list of paths to the run JSON files to merge.
        aggregate_fn: A function that takes a list of individual run results (dicts)
                      and returns either a boolean, single float, or a tuple of 2 floats.
        output_run_id: The ID to use for the generated aggregated run file.
        output_directory: Directory where the aggregated run file should go, if different from the default directory.
        delete_run_files: If True, delete the files specified in run_files after the aggregation file has been created.
    """
    import copy

    if not run_files:
        raise ValueError("No runfiles found!")

    start_time = None
    end_time = None
    first_task_version = None
    first_task_name = None
    first_model_slug = None

    all_individual_results = []

    first_run_data = None
    for run_file in run_files:
        with open(run_file, "r") as f:
            run_data = json.load(f)
            if first_run_data is None:
                first_run_data = run_data

            # Extract the earliest start time and latest end time from the runfiles
            if start_time is None or run_data["startTime"] < start_time:
                start_time = run_data["startTime"]
            if end_time is None or run_data["endTime"] > end_time:
                end_time = run_data["endTime"]

            if first_task_version is None:
                # Extract metadata for validation from the first run
                try:
                    first_task_version = run_data["taskVersion"]
                    first_task_name = first_task_version["name"]
                    first_model_slug = run_data["modelVersion"]["slug"]
                except (KeyError, TypeError) as e:
                    raise ValueError(
                        f"Cannot determine metadata from {run_file}. Malformed data. Error: {e}"
                    )
            else:
                # Validate that essential metadata is consistent across all runs and collect results.
                if run_data["taskVersion"]["name"] != first_task_name:
                    raise ValueError(
                        f"Task name mismatch between {run_files[0]} and {run_file}"
                    )
                if run_data["modelVersion"]["slug"] != first_model_slug:
                    raise ValueError(
                        f"Model version mismatch between {run_files[0]} and {run_file}"
                    )

            # Extract the result dictionary from each run.
            try:
                # This assumes the task returns a dictionary.
                individual_result = run_data["results"][0]
                all_individual_results.append(individual_result)
            except (KeyError, IndexError) as e:
                raise ValueError(
                    f"Could not find 'results' in file {run_file}. Malformed data. Error: {e}"
                )

    if not first_run_data:
        raise ValueError("No run data found in runfiles!")

    aggregated_result = aggregate_fn(all_individual_results)

    # Create the new aggregated run data
    overall_run_data = copy.deepcopy(first_run_data)
    overall_run_data["pyRunId"] = f"{first_task_name}-{output_run_id}"
    overall_run_data["conversations"] = []
    overall_run_data["subruns"] = []
    overall_run_data["assertions"] = []

    match aggregated_result:
        case bool(value):
            overall_run_data["results"] = [
                {"booleanResult": value, "type": "AGGREGATED"}
            ]
        case float(value) | int(value):
            overall_run_data["results"] = [
                {"numericResult": {"value": value}, "type": "AGGREGATED"}
            ]
        case (float(val), float(ci)):
            overall_run_data["results"] = [
                {
                    "numericResult": {
                        "value": val,
                        "confidenceInterval": ci,
                    },
                    "type": "AGGREGATED",
                }
            ]
        case _:
            raise ValueError(
                f"Output of aggregator function is not compatible with booleanResult or numericResult. Output: {aggregated_result}"
            )

    overall_run_data["startTime"] = start_time
    overall_run_data["endTime"] = end_time
    overall_run_data["state"] = "BENCHMARK_TASK_RUN_STATE_COMPLETED"

    if first_task_name is None:
        # This should never happen, but for safety, check anyway
        raise ValueError("Task name for aggregated file is None")

    output_filename = generate_run_filename(first_task_name, output_run_id)

    if output_directory is not None:
        output_filename = Path(output_directory + "/" + output_filename)

    with open(output_filename, "w") as f:
        json.dump(overall_run_data, f, indent=2)

    # Delete a list of files, e.g. if user decides to delete old non-aggregated runfiles after merging
    if delete_run_files:
        for file_path in run_files:
            try:
                Path(file_path).unlink()
            except OSError as e:
                raise OSError(f"Error: {file_path} : {e}")

    return str(output_filename)
