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

"""Tests for the run.json -> harbor converter, written to document it.

SHAPES runs one real kbench task per thing kbench can do -- a tool loop, a
judge, a chat room, a dataset eval, media, an error -- and every shape answers
the same three questions: is the output usable, does it tell the same story as
the run.json, and does it admit what it could not carry. Then one run converted
in full as a worked example, the parts a kbench user would find surprising, and
a tail of degrade paths a real run cannot be asked to produce.

Input is always a run.json a real run wrote, not a hand-written copy of one,
which would keep passing after the serializer changed underneath.
"""

import json
import logging
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from kaggle_benchmarks import (
    ExecutionMode,
    actors,
    assertions,
    chats,
    config,
    kaggle,
    rooms,
    runs,
    task,
)
from kaggle_benchmarks.actors.llms import LLMResponse, OpenAI
from kaggle_benchmarks.content_types import audios, images, videos
from kaggle_benchmarks.kaggle import atif, serialization
from kaggle_benchmarks.llm_messages import LLMMessage
from kaggle_benchmarks.tools.base import ToolInvocation
from kaggle_benchmarks.tools.native import native_tool_agent
from kaggle_benchmarks.ui import ipython_magics
from tests.mocks import MockedChat


@pytest.fixture
def client(monkeypatch):
    """A KaggleClient writing to a temp directory, installed as the global one."""
    with tempfile.TemporaryDirectory() as temp_dir:
        kaggle_client = kaggle.KaggleClient(directory=temp_dir)
        config.execution_mode = ExecutionMode.RUN
        # Pinned, not inherited: Kaggle batch is the mode that writes the
        # run.json files this converter reads, and there a raising task is
        # recorded and moved past. A developer shell defaults to the opposite,
        # so the errored shapes would raise out of the run instead.
        monkeypatch.setattr(config, "continue_with_exceptions", True)
        # Keyed by id(task), and CPython reuses the addresses of collected
        # tasks, which would make run filenames flaky.
        runs._run_counters.clear()
        monkeypatch.setattr("kaggle_benchmarks.client", kaggle_client)
        yield kaggle_client
        config.execution_mode = ExecutionMode.TESTING


@pytest.fixture
def duck():
    """Answers "quack" forever, for runs whose replies do not matter."""
    return MockedChat.from_contents(["quack"], cycle=True, name="Duck")


def _ran(llm, body, name="T", times=1) -> None:
    """Runs `body` as a kbench task, writing one run.json per run."""
    # Built once: runs are numbered per task object, so two objects would write
    # over each other's file.
    built = task(name=name, store_task=False)(body)
    for _ in range(times):
        built.run(*([llm] if llm else []))


def _run_files(client, prefix="") -> list[dict]:
    paths = sorted(Path(client.directory).glob(f"{prefix}*.run.json"))
    return [json.loads(path.read_text()) for path in paths]


def _one(client, prefix="") -> dict:
    (run_json,) = _run_files(client, prefix)
    return run_json


def _messages(steps):
    return [step["message"] for step in steps]


def _warnings(trajectory):
    kbench = trajectory["extra"]["kbench"]
    return [w["kind"] for w in kbench.get("conversion_warnings") or []]


def _every_step(trajectory):
    subs = trajectory.get("subagent_trajectories") or []
    return trajectory["steps"] + [step for sub in subs for step in sub["steps"]]


def add(a: float, b: float) -> float:
    """Adds two numbers."""
    return a + b


def _asks_for(name, arguments, call_id):
    """The reply a backend returns when the model wants a tool run."""
    # MockedChat cannot ask for one, so the call is set the way a backend's
    # normalization sets it.
    message = LLMMessage(sender=None, content="")
    message.tool_calls = [
        ToolInvocation(name=name, arguments=arguments, call_id=call_id)
    ]
    return message


def _judge_llm():
    verdict = {
        "results": [{"criterion": "c", "passed": True, "reason": "y", "confidence": 5}]
    }
    return MockedChat.from_contents_data([verdict], cycle=True, name="Judge")


# --- One shape per thing kbench can do ---


def _plain(duck):
    def plain(llm) -> float:
        llm.prompt("Hello")
        return 0.75

    _ran(duck, plain, "Plain")


def _tool_loop(duck):
    def tools(llm):
        llm.prompt("Hi")
        llm.prompt("Add 2 and 3", tools=[add])
        llm.prompt("Bye")
        return True

    _ran(duck, tools, "Tools")


def _side_chat(duck):
    def aside(llm):
        llm.prompt("Hi")
        with chats.new("Scratchpad"):
            llm.prompt("Think out loud")
        return True

    _ran(duck, aside, "Aside")


def _judged(duck):
    def judged(llm):
        answer = llm.prompt("Capital of France?")
        assertions.assess_response_with_judge(
            ["c"], str(answer), judge_llm=_judge_llm()
        )
        assertions.assert_in("quack", str(answer))
        return True

    _ran(duck, judged, "Judged")


def _chatroom(_duck):
    alice_llm = MockedChat.from_contents(["Ducks."], cycle=True, name="GPTish")
    bob_llm = MockedChat.from_contents(["Geese."], cycle=True, name="Gemish")

    def room_task(llm):
        room = rooms.ChatRoom(system_prompt="Debate the best waterfowl.")
        alice = room.add_participant(alice_llm, name="Alice")
        bob = room.add_participant(bob_llm, name="Bob")
        with room:
            room.post("Topic: ducks versus geese")
            alice.reply()
            bob.reply()
            with room.private_channel([alice], name="Secret") as secret:
                secret.post("Concede gracefully.")
                alice.reply()
        return True

    _ran(alice_llm, room_task, "Room")


_QUESTIONS = pd.DataFrame(
    [
        {"question": "capital of France", "answer": "quack"},
        {"question": "capital of Japan", "answer": "Tokyo"},
        # Half the eval raising is the ordinary case, not a degrade path.
        {"question": "capital of Mars", "answer": None},
    ],
    index=["fr", "jp", "mars"],
)


def _dataset_eval(duck, remove_run_files=False):
    @task(name="row_qa", store_task=False)
    def row_qa(llm, question, answer) -> dict:
        said = str(llm.prompt(question))
        # Not `is None`: pandas 3 stores a missing string as nan, so the row
        # would reach the line below and raise the wrong error.
        if pd.isna(answer):
            raise ValueError("row exploded")
        return {"is_correct": answer.lower() in said.lower()}

    def whole_ds(llm) -> float:
        rows = row_qa.evaluate(
            llm=[llm],
            evaluation_data=_QUESTIONS,
            on_failure="continue",
            remove_run_files=remove_run_files,
        )
        scored = [row.result["is_correct"] for row in rows if row.result]
        return sum(scored) / len(rows)

    _ran(duck, whole_ds, "whole_ds")


# One red pixel: the smallest thing that is really a png.
RED_DOT = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAh"
    "KmMIQAAAABJRU5ErkJggg=="
)


def _with_media(_duck):
    llm = MockedChat.from_contents(["Seen."], cycle=True, name="Duck")

    def media_task(llm):
        llm.prompt("What colour?", image=images.from_base64(RED_DOT, format="png"))
        # Sent directly: prompt(image=ImageURL) downloads and inlines it.
        actors.user.send(images.from_url("https://example.test/logo.png"))
        llm.prompt("What does it say?")
        # Unpadded: 3.12's base64 rejects "//uQ==" as over-padded, and the
        # helper raises, so the rest of this task would never run.
        llm.prompt("Transcribe.", audio=audios.from_base64("//uQ", format="mp3"))
        # YouTube because kbench accepts no other video url.
        clip = videos.from_url("https://www.youtube.com/watch?v=aqz-KE-bpKQ")
        llm.prompt("Describe.", video=clip)
        return True

    _ran(llm, media_task, "Media")


def _errored(duck):
    def boom(llm):
        llm.prompt("Hi")
        raise ValueError("kaboom")

    _ran(duck, boom, "Boom")


def _no_model_at_all(_duck):
    def mute() -> bool:
        return True

    _ran(None, mute, "Mute")


SHAPES = [
    _plain,
    _tool_loop,
    _side_chat,
    _judged,
    _chatroom,
    _dataset_eval,
    _with_media,
    _errored,
    _no_model_at_all,
]
shapes = pytest.mark.parametrize("shape", SHAPES, ids=lambda s: s.__name__.strip("_"))

# Only the 3.12 CI leg installs harbor: it is a heavy test-only dependency, and
# 3.11 cannot import the version that knows this schema. Everywhere else the
# check below skips, and the assertions above still say whether output is right.
try:
    from harbor.models.trajectories.trajectory import Trajectory
    from harbor.models.trial.result import TrialResult
except ImportError:
    Trajectory = TrialResult = None

needs_harbor = pytest.mark.skipif(Trajectory is None, reason="harbor not installed")


@needs_harbor
@shapes
def test_every_shape_writes_files_harbor_accepts(client, duck, shape):
    """The only check that catches "we produced something harbor rejects". Reads
    what landed on disk, not what the converter returned, so a wire-up that
    passed the wrong arguments fails here too. ATIF forbids unknown keys, so a
    typo is an error and not a field that quietly vanishes."""
    shape(duck)
    written = sorted(Path(client.directory).glob("*.atif.json"))
    assert written
    for path in written:
        Trajectory(**json.loads(path.read_text()))
        result = path.with_name(path.name.replace(".atif.", ".result."))
        TrialResult(**json.loads(result.read_text()))


@shapes
def test_every_shape_produces_a_usable_trajectory(client, duck, shape):
    """Schema-valid is not usable. These promises hold whatever the run did."""
    shape(duck)
    for run_json in _run_files(client):
        trajectory = atif.to_atif(run_json)
        result = atif.to_trial_result(run_json)
        # Something to read, always: harbor rejects a trajectory with no step.
        assert trajectory["steps"]
        subagents = trajectory.get("subagent_trajectories") or []
        for steps in [trajectory["steps"]] + [sub["steps"] for sub in subagents]:
            assert [s["step_id"] for s in steps] == list(range(1, len(steps) + 1))
        # Harbor rejects the whole file over one of these on a non-agent step.
        for step in _every_step(trajectory):
            if step["source"] != "agent":
                assert not {"model_name", "metrics", "reasoning_content"} & set(step)
        # Every ref resolves in-file or at a path the caller named.
        embedded = {sub["trajectory_id"] for sub in subagents}
        for step in trajectory["steps"]:
            for res in (step.get("observation") or {}).get("results") or []:
                for ref in res.get("subagent_trajectory_ref") or []:
                    assert ref.get("trajectory_id", "") in embedded or "trajectory_path"
        # Harbor reads an empty container as absent, and an explicit one reads
        # as measured-and-nothing.
        _no_empties(trajectory)
        _no_empties(result)
        # No uuid4 and no now(), so the C# backfill can be diffed against this.
        assert atif.to_atif(run_json) == trajectory
        assert atif.to_trial_result(run_json) == result
        assert result["task_name"] == run_json["taskVersion"]["name"]


def _no_empties(node, path="$"):
    if isinstance(node, dict):
        for key, value in node.items():
            # Required by harbor even when the tool takes no arguments.
            assert key == "arguments" or value not in (None, {}, []), f"{path}.{key}"
            _no_empties(value, f"{path}.{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _no_empties(value, f"{path}[{index}]")


@shapes
def test_every_shape_tells_the_same_story_as_its_run_json(client, duck, shape):
    """Nothing kbench recorded goes missing, and nothing is invented."""
    # By content, not position: a tool loop is spliced back where it branched
    # and a side chat becomes a subagent. Tool blobs are excluded, since they
    # leave the transcript for tool_calls and observation. How often a message
    # appears is pinned per shape below, since a fork copies its parent's.
    shape(duck)
    for run_json in _run_files(client):
        written = {
            part.get("text")
            for conversation in run_json["conversations"]
            for request in conversation.get("requests") or []
            for content in request.get("contents") or []
            if content.get("role") != atif.TOOL_ROLE
            for part in content.get("parts") or []
            if part.get("text")
        }
        converted = {
            text
            for step in _every_step(atif.to_atif(run_json))
            for text in (
                [step["message"]]
                if isinstance(step["message"], str)
                else [part.get("text") for part in step["message"]]
            )
            if text
        }
        assert not written - converted, "a message kbench recorded went missing"
        # What the converter added is only ever something it marked as its own.
        markers = (atif.PLACEHOLDER, atif.ERROR, atif.DELEGATED)
        for text in converted - written:
            assert "not representable in ATIF" in text or text.startswith(markers)


@shapes
def test_every_shape_is_honest_about_what_it_could_not_carry(client, duck, shape):
    """A warning records a loss in the file, not just the log, which is gone
    once the notebook closes. Any other warning is a regression."""
    expected = {
        _with_media: {"media_unrepresentable"},
        # This caller did not name the rows' files, and the parent cannot.
        _dataset_eval: {"subrun_path_unknown"},
    }
    shape(duck)
    reported = set()
    for run_json in _run_files(client):
        trajectory = atif.to_atif(run_json)
        reported |= set(_warnings(trajectory))
        # A reader can always get back to the original, name for name.
        carried = trajectory["extra"]["kbench"]["task_version"]
        assert carried == atif._snake_keys(run_json["taskVersion"])
    assert reported == expected.get(shape, set())


def test_carried_fields_are_renamed_all_the_way_down_but_user_keys_are_not(client):
    """`MessageToJson` camelCases the whole run.json, so undoing it only at the
    top level would leave two spellings nested inside one another. Rewriting a
    task author's own dict keys would be the opposite mistake: those are data,
    not proto fields, and nothing else could put them back."""
    run_json = {
        "pyRunId": "X-Run #1",
        "taskVersion": {"name": "X", "versionNumber": 2},
        "results": [{"type": "AGGREGATED", "dictResult": {"myScore": 1.0}}],
    }
    trajectory = atif.to_atif(run_json)
    assert trajectory["extra"]["kbench"]["task_version"] == {
        "name": "X",
        "version_number": 2,
    }
    assert trajectory["final_metrics"]["extra"]["kbench_result"] == {"myScore": 1.0}


# --- What the conversion looks like ---


def test_a_converted_run_in_full(client):
    """One ordinary run converted whole, so the mapping can be read in one place
    rather than assembled out of the single-aspect assertions below."""
    llm = MockedChat(
        responses=[
            LLMMessage(sender=None, content="Hello there."),
            _asks_for("add", {"a": 2, "b": 3}, "call_A"),
            LLMMessage(sender=None, content="The answer is 5."),
        ],
        name="Duck",
    )

    def demo(llm) -> float:
        llm.prompt("Hello")
        llm.prompt("Add 2 and 3", tools=[add])
        return 0.75

    _ran(llm, demo, "Demo")
    run_json = _one(client)
    trajectory = atif.to_atif(run_json)
    # taskVersion carries the task's whole source, which would make this test
    # depend on its own formatting.
    assert trajectory["extra"]["kbench"].pop("task_version")["name"] == "Demo"
    agent = {"name": "kaggle-benchmarks", "version": atif._harness_version()}
    assert trajectory == {
        "schema_version": "ATIF-v1.7",
        "session_id": "Demo-Run #1",
        "trajectory_id": "Demo-Run #1",
        "agent": {**agent, "model_name": "Duck"},
        "steps": [
            {
                "step_id": 1,
                "source": "user",
                "message": "Hello",
                "extra": {"sender_name": "User"},
                # Only the first: run.json times the run, not each message.
                "timestamp": run_json["startTime"],
            },
            {
                "step_id": 2,
                "source": "agent",
                "message": "Hello there.",
                "extra": {"sender_name": "Duck"},
                "model_name": "Duck",
            },
            {
                "step_id": 3,
                "source": "user",
                "message": "Add 2 and 3",
                "extra": {"sender_name": "User"},
            },
            {
                # The result folded onto the call, so the pair is one step.
                "step_id": 4,
                "source": "agent",
                "message": "",
                "extra": {"sender_name": "Duck"},
                "model_name": "Duck",
                "tool_calls": [
                    {
                        "tool_call_id": "call_A",
                        "function_name": "add",
                        "arguments": {"a": 2, "b": 3},
                    }
                ],
                "observation": {
                    "results": [{"source_call_id": "call_A", "content": "5"}]
                },
            },
            {
                "step_id": 5,
                "source": "agent",
                "message": "The answer is 5.",
                "extra": {"sender_name": "Duck"},
                "model_name": "Duck",
            },
        ],
        "notes": "Converted from kbench run.json for task 'Demo'.",
        # ATIF has no reward field anywhere, so the score rides in extra while
        # result.json keeps the authoritative copy.
        "final_metrics": {
            "total_steps": 5,
            "extra": {"kbench_result": {"score": 0.75}},
        },
        "extra": {
            "kbench": {
                "state": "BENCHMARK_TASK_RUN_STATE_COMPLETED",
                "end_time": run_json["endTime"],
            }
        },
    }
    assert atif.to_trial_result(run_json) == {
        "task_name": "Demo",
        "trial_name": "Demo-Run #1",
        # Deliberately implausible: anything path-shaped invites someone to
        # treat a converted run as a real harbor trial.
        "trial_uri": "kbench://not-a-harbor-trial",
        "task_id": {"path": "kbench://not-a-harbor-task"},
        "task_checksum": "unknown (kbench)",
        "config": {"task": {"path": "kbench://not-a-harbor-task"}},
        "agent_info": {**agent, "model_info": {"name": "Duck"}},
        "verifier_result": {"rewards": {"score": 0.75}},
        "started_at": run_json["startTime"],
        "finished_at": run_json["endTime"],
        # No id: harbor defaults it to a fresh uuid4, so reconverting the same
        # input would otherwise produce a different file.
    }


# --- Surprises, for someone reading a run.json beside its trajectory ---


def test_a_tool_loop_stays_in_the_conversation_it_runs_in(client, duck):
    """The loop builds each round's view locally instead of forking, so its turns
    are already in order in the one conversation kbench recorded."""
    _tool_loop(duck)
    run_json = _one(client)
    (conversation,) = run_json["conversations"]
    said = ["Hi", "quack", "Add 2 and 3", "quack", "Bye", "quack"]
    assert _messages(atif.to_atif(run_json)["steps"]) == said


def test_a_recorded_tool_loop_fork_is_spliced_where_it_branched(client, duck):
    """Runs recorded before the loop stopped forking still hold two overlapping
    conversations. Spliced where it branched, not appended, so the fork does not
    read as a second agent that spoke after the fact."""

    def forked(llm):
        llm.prompt("Hi")
        with chats.fork(name="Tool loop"):
            llm.prompt("Add 2 and 3")
        llm.prompt("Bye")
        return True

    _ran(duck, forked, "Tools")
    run_json = _one(client)
    assert len(run_json["conversations"]) == 2
    # "Add 2 and 3" lands in the middle, and the shared "Hi"/"quack" is not
    # repeated: the fork copied it, and the merge drops the copy.
    said = ["Hi", "quack", "Add 2 and 3", "quack", "Bye", "quack"]
    assert _messages(atif.to_atif(run_json)["steps"]) == said


@pytest.mark.parametrize("branch", [chats.new, chats.fork])
def test_any_other_branch_becomes_a_separate_agent(client, duck, branch):
    """Only a chat named "Tool loop" is merged back, which is a decision on the
    name and not on overlap: a fork copies its parent's history, so from outside
    it looks exactly like a tool loop."""

    def side(llm):
        llm.prompt("Hi")
        with branch("Aside"):
            llm.prompt("Separately")
        return True

    _ran(duck, side, "Side")
    trajectory = atif.to_atif(_one(client))
    (subagent,) = trajectory["subagent_trajectories"]
    # The turns, then the step naming what ran beside them.
    assert _messages(trajectory["steps"]) == [
        "Hi",
        "quack",
        f"{atif.DELEGATED} Other trajectories from this run: Aside.",
    ]
    assert subagent["agent"]["name"] == "Aside"
    assert "Separately" in _messages(subagent["steps"])


def test_a_judge_is_named_for_the_model_that_graded(client, duck):
    """The run's model_name is the model being graded, so the judge's own would be
    lost; the chat's name is the only place the file records it."""
    _judged(duck)
    trajectory = atif.to_atif(_one(client))
    (subagent,) = trajectory["subagent_trajectories"]
    assert trajectory["agent"]["model_name"] == "Duck"
    # Harbor validates a subagent as a trajectory in its own right, so it
    # carries its own schema_version and numbers its steps from 1.
    assert subagent["schema_version"] == atif.SCHEMA_VERSION
    version = atif._harness_version()
    assert subagent["agent"] == {
        "name": "judge",
        "version": version,
        "model_name": "Judge",
    }
    assert [step["step_id"] for step in subagent["steps"]] == [1, 2]


def test_a_room_full_of_models_is_one_subagent(client, duck):
    """A room is one conversation with its speakers in senderName and no
    per-participant model, so splitting per participant would write one model on
    both. Its private channel is a second conversation, so the visibility
    boundary survives without being reconstructed."""
    _chatroom(duck)
    run_json = _one(client)
    subagents = atif.to_atif(run_json)["subagent_trajectories"]
    assert run_json["modelVersion"]["slug"] == "GPTish"
    assert [sub["agent"]["name"] for sub in subagents] == ["Narrator", "Secret"]
    spoke = [step["extra"]["sender_name"] for sub in subagents for step in sub["steps"]]
    assert spoke == ["Narrator", "Alice", "Bob", "Secret", "Alice"]
    # kbench flattens a channel into the same list as the room that owns it, so
    # both are siblings by the time we see them and one step names them both.
    trajectory = atif.to_atif(run_json)
    assert _messages(trajectory["steps"])[-1] == (
        f"{atif.DELEGATED} Other trajectories from this run: Narrator, Secret."
    )
    refs = trajectory["steps"][-1]["observation"]["results"][0]
    assert [ref["extra"]["kbench_chat"] for ref in refs["subagent_trajectory_ref"]] == [
        "Narrator",
        "Secret",
    ]
    # No subagent carries refs of its own: the nesting is gone before we see it.
    assert not any(step.get("observation") for s in subagents for step in s["steps"])


def test_a_dataset_evals_rows_are_siblings_the_parent_cannot_name(client, duck):
    """An eval writes N+1 files and the parent cannot point at the N: a row's
    filename comes from its cache_id, and nothing in the embedded subrun records
    it. The caller must supply the names, because a ref to a file that does not
    exist is worse than no ref at all."""
    _dataset_eval(duck)
    rows = _run_files(client, "row_qa")
    parent = _one(client, "whole_ds")
    ids = ["row_qa-Run #1", "row_qa-Run #2", "row_qa-Run #3"]
    assert sorted(p.name for p in Path(client.directory).glob("row_qa*.run.json")) == [
        f"row_qa-run_param_id_{label}_Duck.run.json" for label in ("fr", "jp", "mars")
    ]
    assert [s["pyRunId"] for s in parent["subruns"]] == ids
    assert atif.to_atif(parent)["extra"]["kbench"]["conversion_warnings"] == [
        {"kind": "subrun_path_unknown", "detail": row_id} for row_id in ids
    ]
    # Given the names from outside, the parent's last step points at the rows.
    paths = {row_id: f"{row_id}.atif.json" for row_id in ids}
    steps = atif.to_atif(parent, subrun_paths=paths)["steps"]
    (refs,) = steps[-1]["observation"]["results"]
    found = {r["trajectory_path"] for r in refs["subagent_trajectory_ref"]}
    assert found == set(paths.values())
    # Grouped by source, the way harbor groups a job's trials.
    assert {atif.to_trial_result(r, source="ds")["source"] for r in rows} == {"ds"}


def test_a_row_that_failed_is_summarised_beside_one_that_did_not(client, duck):
    """The parent embeds each row's outcome, so an eval where one row raised does
    not read like one where none did."""
    _dataset_eval(duck)
    right, wrong, bad = atif.to_atif(_one(client, "whole_ds"))["extra"]["kbench"][
        "subruns"
    ]
    assert right["rewards"] == {"is_correct": 1.0}
    assert wrong["rewards"] == {"is_correct": 0.0}
    assert right["state"] == wrong["state"] == "BENCHMARK_TASK_RUN_STATE_COMPLETED"
    assert bad["state"] == "BENCHMARK_TASK_RUN_STATE_ERRORED"
    assert "row exploded" in bad["error_message"]
    # A row that raised returned nothing, so kbench scores run.passed instead:
    # the 0.0 is "this row failed", not "this row answered wrongly".
    assert bad["rewards"] == {"score": 0.0}


def test_a_cached_row_keeps_its_score_but_loses_its_transcript(
    client, monkeypatch, duck
):
    """Rerunning an eval from cache restores only each row's result, so the
    embedded subruns have no conversation and no times. Nothing is lost: the
    row's own file still holds the transcript, and the parent's summary is built
    from fields caching does restore, so it is identical either way."""
    _dataset_eval(duck)
    fresh = _one(client, "whole_ds")
    # Only the parent is removed, so the rows are served from their files.
    next(Path(client.directory).glob("whole_ds*.run.json")).unlink()
    cached_client = kaggle.KaggleClient(directory=client.directory, use_cache=True)
    monkeypatch.setattr("kaggle_benchmarks.client", cached_client)
    runs._run_counters.clear()
    _dataset_eval(duck)
    cached = _one(client, "whole_ds")
    assert [len(row.get("conversations") or []) for row in fresh["subruns"]] == [
        1,
        1,
        1,
    ]
    # The row that raised is not cached, so it ran again and has its transcript.
    assert [len(row.get("conversations") or []) for row in cached["subruns"]] == [
        0,
        0,
        1,
    ]
    assert [("startTime" in row) for row in cached["subruns"]] == [False, False, True]
    assert (
        atif.to_atif(cached)["extra"]["kbench"]["subruns"]
        == atif.to_atif(fresh)["extra"]["kbench"]["subruns"]
    )


def test_an_aggregate_has_no_transcript_and_says_how_many_it_merged(client, duck):
    """Merging repeat runs drops their conversations, so an aggregate is the only
    run.json with no conversation at all -- a run that merely said nothing still
    has an empty one."""

    def rep(llm) -> float:
        llm.prompt("Hi")
        return 0.5

    _ran(duck, rep, "Rep", times=2)
    output = serialization.merge_results_from_runfiles(
        sorted(str(p) for p in Path(client.directory).glob("Rep*.run.json")),
        lambda rs: sum(r["numericResult"].get("value", 0.0) for r in rs) / len(rs),
        output_directory=str(client.directory),
    )
    aggregate = json.loads(Path(output).read_text())
    trajectory = atif.to_atif(aggregate)
    # The stubs are the only record of the inputs once their files are deleted.
    assert [stub["pyRunId"] for stub in aggregate["subruns"]] == [
        "Rep-Run #1",
        "Rep-Run #2",
    ]
    assert all("conversations" not in stub for stub in aggregate["subruns"])
    said = "[placeholder] Aggregated from 2 runs. No conversation recorded."
    assert _messages(trajectory["steps"]) == [said]
    # 0 while one step exists: the step is a stand-in, and counting it would
    # report a turn that never happened.
    assert trajectory["final_metrics"]["total_steps"] == 0
    assert "placeholder" in trajectory["notes"]
    assert atif.to_trial_result(aggregate)["verifier_result"] == {
        "rewards": {"score": 0.5}
    }


def test_an_image_by_url_is_the_one_media_atif_can_point_at(client, duck):
    """A type ATIF names, in the shape it stores: ImageSource is a media_type and
    a path, so a url fits and nothing is lost. The message becomes a list of parts
    rather than a string, which is harbor's own rule for content that is not only
    text, and is why a list here means a picture."""
    _with_media(duck)
    trajectory = atif.to_atif(_one(client))
    shown = [
        part
        for message in _messages(trajectory["steps"])
        if not isinstance(message, str)
        for part in message
        if part["type"] != "text"
    ]
    assert shown == [
        {
            "type": "image",
            "source": {
                "media_type": "image/png",
                "path": "https://example.test/logo.png",
            },
        }
    ]


def test_media_atif_cannot_hold_is_marked_in_place_and_kept_in_extra(client, duck):
    """Three ways to miss, and the payload survives all of them. An inline image is
    a type ATIF names in a shape it has no field for; audio and video are types it
    cannot name at all -- v1.6 added images and v1.8 audio, and we emit v1.7. That
    the video arrives as a url and is still unrepresentable is what separates the
    two: for it the type is the blocker, not the shape."""
    _with_media(duck)
    trajectory = atif.to_atif(_one(client))
    marked = [
        message
        for message in _messages(trajectory["steps"])
        if isinstance(message, str) and "not representable in ATIF" in message
    ]
    assert marked == [
        "[image/png: inline data -- not representable in ATIF]",
        "[audio/mp3: inline data -- not representable in ATIF]",
        "[video/*: https://www.youtube.com/watch?v=aqz-KE-bpKQ"
        " -- not representable in ATIF]",
    ]
    # Lossy in the message, lossless in the file.
    carried = [
        media
        for step in trajectory["steps"]
        for media in step["extra"].get("kbench_media") or []
    ]
    assert {m["mime_type"] for m in carried} == {"image/png", "audio/mp3", "video/*"}
    assert [m["kind"] for m in carried].count("inline_data") == 2
    # What a machine counts, rather than the message text.
    kinds = [w["kind"] for w in trajectory["extra"]["kbench"]["conversion_warnings"]]
    assert kinds == ["media_unrepresentable"] * 3


def test_assertions_survive_but_their_pointers_do_not(client, duck):
    """ATIF has no assertion of its own, so kbench's are carried whole -- names
    renamed, structure untouched -- and each names the request it was checked
    against, which resolves to nothing here, since a request boundary is a
    serializer grouping the converter flattens away. The dangling pointer is
    kept rather than pruned, so it is visibly kbench's."""
    _judged(duck)
    run_json = _one(client)
    carried = atif.to_atif(run_json)["extra"]["kbench"]["assertions"]
    assert carried == atif._snake_keys(run_json["assertions"])
    assert {e["status"] for e in carried} == {
        "BENCHMARK_TASK_RUN_ASSERTION_STATUS_PASSED"
    }
    named = {
        ids["request_id"] for e in carried for ids in e["conversation_request_ids"]
    }
    assert named and not named & {"step_id", "request_id"}


@pytest.mark.parametrize(
    "returned, rewards",
    [
        (True, {"score": 1.0}),
        (False, {"score": 0.0}),
        (0.75, {"score": 0.75}),
        # proto3 omits zero-valued scalars, so this arrives as `{}`. Read as "no
        # score" instead, a failing run would look unscored.
        (0.0, {"score": 0.0}),
        # PassCount and MetricWithCI both serialize as a numeric result.
        ((0.6, 0.05), {"score": 0.6, "score_confidence_interval": 0.05}),
        # bool is an int subclass, so leaf handling has to decide about it.
        ({"is_correct": True, "n": 3}, {"is_correct": 1.0, "n": 3.0}),
        # rewards maps a name to a number, so a gold answer has nothing to put
        # there. An empty map would claim a verifier ran and scored nothing, so
        # the object is omitted: the midtier reads absent as unscored.
        ({"gold": "Tokyo"}, None),
        # No result is not unscored: kbench serializes run.passed instead, and a
        # run that did not raise passed.
        (None, {"score": 1.0}),
    ],
)
def test_every_result_type_a_task_can_return(client, duck, returned, rewards):
    def result_task(llm):
        llm.prompt("Hi")
        return returned

    _ran(duck, result_task, "Result")
    run_json = _one(client)
    trajectory = atif.to_atif(run_json)
    if rewards is None:
        assert "verifier_result" not in atif.to_trial_result(run_json)
        assert "extra" not in trajectory["final_metrics"]
    else:
        assert atif.to_trial_result(run_json)["verifier_result"]["rewards"] == rewards
        # The same numbers ride in the trajectory, where a reader of the
        # transcript alone can see them.
        assert trajectory["final_metrics"]["extra"]["kbench_result"] == rewards


def test_a_failing_assertion_scores_zero_without_an_exception(client, duck):
    """The idiomatic kbench task returns nothing and asserts instead: PassFail
    serializes run.passed, which a failing assertion makes false. So a run can
    score 0 having raised nothing, and exception_info is what tells the two
    apart -- harbor's analyzer reads a set one as failed whatever the reward."""

    def checked(llm):
        llm.prompt("Hi")
        assertions.assert_true(False, expectation="deliberately failing")

    _ran(duck, checked, "Checked")
    run_json = _one(client)
    result = atif.to_trial_result(run_json)
    assert result["verifier_result"]["rewards"] == {"score": 0.0}
    assert "exception_info" not in result
    # The transcript is of a run that went fine; only the assertion says not.
    assert _messages(atif.to_atif(run_json)["steps"]) == ["Hi", "quack"]
    (carried,) = atif.to_atif(run_json)["extra"]["kbench"]["assertions"]
    assert carried["status"].endswith("FAILED")


def test_an_errored_run_keeps_its_transcript_and_its_score(client, duck):
    """kbench records a failure and returns rather than re-raising, so an errored
    run still has its transcript and can still have been scored. Both are keyed
    off errorMessage, never state, or a run that raised after scoring would lose
    the score."""
    _errored(duck)
    run_json = _one(client)
    run_json["results"] = [{"numericResult": {"value": 0.4}, "type": "AGGREGATED"}]
    result = atif.to_trial_result(run_json)
    # The midtier renders "{type}: {message}" to a user, so the traceback goes
    # in the field named for it rather than into the message.
    assert result["exception_info"]["exception_type"] == "ValueError"
    assert result["exception_info"]["exception_message"] == "kaboom"
    assert "Traceback" in result["exception_info"]["exception_traceback"]
    assert result["verifier_result"] == {"rewards": {"score": 0.4}}
    # The turns survive, and a closing step says how they stopped: ATIF has no
    # field for a failure, so the transcript would otherwise end mid-air.
    trajectory = atif.to_atif(run_json)
    assert _messages(trajectory["steps"]) == [
        "Hi",
        "quack",
        f"{atif.ERROR} Run failed after the last model turn: ValueError: kaboom",
    ]
    # Of what the model did, so the step kbench added is not one of them.
    assert trajectory["final_metrics"]["total_steps"] == 2


def test_a_loop_out_of_rounds_says_its_transcript_is_missing_the_end(client):
    """The loop raises instead of replying and kbench keeps nothing past the last
    assistant message, so the closing turns are gone before the converter sees
    the file. It cannot recover them, so it says so."""

    def greedy_task(llm):
        native_tool_agent(llm, tools=[add], max_tool_rounds=2)
        return True

    calls = [_asks_for("add", {"a": 1, "b": 1}, "c") for _ in range(10)]
    _ran(MockedChat(responses=calls, name="Duck"), greedy_task, "Greedy")
    run_json = _one(client)
    assert "transcript_truncated_by_serializer" in _warnings(atif.to_atif(run_json))
    failed = atif.to_trial_result(run_json)["exception_info"]
    assert failed["exception_type"] == "ToolInvocationLimitExhausted"


def test_a_message_after_the_last_reply_is_still_a_step(client, duck):
    """Requests are flushed at an assistant reply, so anything said afterwards has
    no reply to close it. It still happened, so it is still a turn."""

    def trailing(llm):
        llm.prompt("Hi")
        actors.user.send("Thanks, bye")
        return True

    _ran(duck, trailing, "Trailing")
    assert _messages(atif.to_atif(_one(client))["steps"]) == [
        "Hi",
        "quack",
        "Thanks, bye",
    ]


def test_the_models_reasoning_reaches_the_step_it_explains(client):
    """reasoning_content is a property of the turn and only an agent step may
    carry it, so it rides on the reply that closed the request."""

    class Reasoner(OpenAI):
        def _call_api(self, messages, **kwargs):
            return LLMResponse(content="quack", reasoning_traces="First, ducks.")

    def reasoned(llm):
        llm.prompt("Hi")
        return True

    _ran(
        Reasoner(client=None, model="mock-reasoning", name="Duck"), reasoned, "Reasoned"
    )
    steps = atif.to_atif(_one(client))["steps"]
    assert "reasoning_content" not in steps[0]
    assert steps[1]["reasoning_content"] == "First, ducks."


def test_what_a_run_spent_is_totalled_once_despite_the_fork(client, monkeypatch):
    """Per-step metrics and the run total come from the same messages, so the fork
    -- which copies its parent's requests, metrics included -- cannot inflate one
    without inflating the other."""
    respond = actors.LLMChat.respond

    def stamped(self, *args, **kwargs):
        message = respond(self, *args, **kwargs)
        message._meta = {"input_tokens": 100, "output_tokens": 10}
        return message

    monkeypatch.setattr(actors.LLMChat, "respond", stamped)
    _tool_loop(MockedChat.from_contents(["a", "b", "c"], name="Duck"))
    run_json = _one(client)
    # Three replies were produced, so three lots of tokens were spent. Counting
    # the fork's copy of the first would give 400.
    steps = atif.to_atif(run_json)["steps"]
    assert sum(s.get("metrics", {}).get("prompt_tokens", 0) for s in steps) == 300
    spent = {"n_input_tokens": 300, "n_output_tokens": 30}
    assert atif.to_trial_result(run_json)["agent_result"] == spent
    # A run nobody priced reports no cost rather than a measured zero.
    assert "total_cost_usd" not in atif.to_atif(run_json)["final_metrics"]


def test_one_odd_attribute_does_not_lose_the_whole_run_file(client, duck):
    """A message holding an object with an unserializable attribute used to raise
    while the file was being written, so the run vanished."""

    class Weird:
        def __init__(self):
            self.duck = object()

    def weird(llm):
        actors.user.send(Weird())
        llm.prompt("Hi")
        return True

    with pytest.warns(UserWarning, match="lacks a proper serialization method"):
        _ran(duck, weird, "Weird")
    messages = _messages(atif.to_atif(_one(client))["steps"])
    # The attribute degrades to its string form; everything else is intact.
    assert "<object object at" in messages[0]
    assert messages[1:] == ["Hi", "quack"]


# --- Degrade paths a real run cannot be asked for ---
#
# Hand-written run.json fragments, because kbench does not produce an unknown
# content role or a chained traceback on request.

ASSISTANT, USER = atif.ASSISTANT_ROLE, "CONTENT_ROLE_USER"


def _said(role, text="hi", sender="X"):
    return {"role": role, "senderName": sender, "parts": [{"text": text}]}


def _tool(blob):
    # Encoded twice, the way the serializer writes a tool result.
    text = json.dumps(json.dumps(blob))
    return {"role": atif.TOOL_ROLE, "senderName": "t", "parts": [{"text": text}]}


def _chat(*contents, chat_id="T-00000001", metrics=None):
    request = {"id": f"{chat_id}-req-1", "contents": list(contents)}
    if metrics:
        request["metrics"] = metrics
    return {"id": chat_id, "requests": [request]}


def _run(*contents, **overrides):
    """A minimal run.json holding one chat. A field set to None is removed."""
    base = {
        "pyRunId": "Task-Run #1",
        "taskVersion": {"name": "Task"},
        "modelVersion": {"slug": "Duck"},
        "startTime": "2026-01-01T00:00:00Z",
        "endTime": "2026-01-01T00:00:01Z",
        "conversations": [_chat(*contents)] if contents else [],
        "results": [],
        **overrides,
    }
    return {key: value for key, value in base.items() if value is not None}


@pytest.mark.parametrize("convert", [atif.to_atif, atif.to_trial_result])
def test_a_run_without_a_task_name_cannot_be_converted(convert):
    """The one failure the converter raises on: harbor requires task_name and
    there is nothing to fall back to."""
    with pytest.raises(atif.ConversionError, match="taskVersion.name"):
        convert(_run(taskVersion=None))


@pytest.mark.parametrize(
    "run_json, expected",
    [
        ({"pyRunId": "T-Run #1", "id": "123"}, {"trial_name": "T-Run #1"}),
        ({"id": "123"}, {"trial_name": "123"}),
        ({}, {"trial_name": "unknown"}),
        # ModelInfo.name is required, so a nameless model has no object to
        # write. Unlike trial_uri nothing requires model_info, so none is made.
        ({"modelVersion": {}}, {"agent_info": {"model_info": None}}),
        # Harbor's JobStats keys on "{agent}__{model}__{source}", so an eval's
        # rows and their parent have to agree on it.
        ({"_source": "whole_ds"}, {"source": "whole_ds"}),
        ({}, {"source": None}),
    ],
)
def test_what_a_trial_result_is_named_when_the_run_names_little(run_json, expected):
    source = run_json.pop("_source", None)
    result = atif.to_trial_result(
        {"taskVersion": {"name": "T"}, **run_json}, source=source
    )
    for field, value in expected.items():
        if value is None:
            assert field not in result
        elif isinstance(value, dict):
            assert not set(value) & set(result[field])
        else:
            assert result[field] == value


@pytest.mark.parametrize(
    "content, kbench_role, warned",
    [
        # ATIF has three sources, so these collapse onto system and record the
        # original, which keeps a developer turn tellable from a system one.
        (_said("CONTENT_ROLE_DEVELOPER"), "CONTENT_ROLE_DEVELOPER", []),
        # Every role="tool" actor lands on CONTEXT, so a tool result shares its
        # role with genuine context.
        (_said(atif.TOOL_ROLE), atif.TOOL_ROLE, []),
        (_said("CONTENT_ROLE_SYSTEM"), None, []),
        # The proto only ever grows, so a role we have never seen is a newer
        # kbench, not a broken file. It lands on the source claiming the least.
        (_said("CONTENT_ROLE_FUTURE"), None, ["unknown_content_role"]),
        # proto3 omits the default, so UNSPECIFIED arrives as an absent key.
        ({"parts": [{"text": "hi"}]}, None, ["unknown_content_role"]),
    ],
)
def test_the_roles_that_are_not_user_or_assistant_become_system(
    content, kbench_role, warned
):
    trajectory = atif.to_atif(_run(content))
    (step,) = trajectory["steps"]
    assert step["source"] == "system"
    assert step.get("extra", {}).get("kbench_role") == kbench_role
    assert _warnings(trajectory) == warned


def test_a_fork_sharing_no_history_is_kept_separate():
    """Named like a tool loop but sharing no prefix, which fork() cannot produce.
    Merging would interleave two unrelated threads."""
    trajectory = atif.to_atif(
        _run(
            conversations=[
                _chat(_said(USER, "Hi"), chat_id="Main-00000001"),
                _chat(_said(USER, "Unrelated"), chat_id="Tool loop-00000002"),
            ]
        )
    )
    assert _messages(trajectory["steps"])[0] == "Hi"
    assert _warnings(trajectory) == ["fork_without_common_prefix"]
    assert len(trajectory["subagent_trajectories"]) == 1


def test_a_tool_result_with_no_call_before_it_stays_a_step():
    """tool_calls is agent-only, so hanging one off a user step would make harbor
    reject the whole trajectory."""
    trajectory = atif.to_atif(
        _run(_said(USER), _tool({"name": "add", "arguments": {}}))
    )
    assert [step["source"] for step in trajectory["steps"]] == ["user", "system"]
    assert _warnings(trajectory) == ["tool_result_without_call"]


def test_a_tool_that_refused_or_failed_says_so_on_the_result():
    """invoke_tool keeps arguments it could not parse as a string and refuses the
    call, so the pair is real but arguments has no dict to offer. A tool that
    raised still produced an observation, so the failure goes on the result
    rather than reading as an ordinary return value."""
    refused = {
        "name": "add",
        "arguments": "{oops",
        "call_id": "c1",
        "error": "Error: Tool 'add' arguments could not be parsed as JSON",
    }
    raised = {
        "name": "f",
        "arguments": {},
        "call_id": "c2",
        "error": "RuntimeError: boom",
    }
    trajectory = atif.to_atif(_run(_said(ASSISTANT, ""), _tool(refused), _tool(raised)))
    (step,) = trajectory["steps"]
    assert step["tool_calls"][0]["arguments"] == {}
    assert _warnings(trajectory) == ["tool_arguments_unparsed"]
    # The unparsed original is in the error the tool returned, so it is not lost.
    assert "could not be parsed" in step["observation"]["results"][0]["content"]
    assert step["observation"]["results"][1] == {
        "source_call_id": "c2",
        "content": "RuntimeError: boom",
        "extra": {"is_error": True},
    }


def test_a_backend_that_returns_no_call_id_still_gets_linkable_pairs():
    """ToolCall.tool_call_id is required and ToolInvocation.call_id is not. One id
    reused would make every result look like the answer to the same call."""
    idless = _tool({"name": "now", "arguments": {}, "output": "12:00"})
    run = _run(_said(ASSISTANT, ""), idless, idless, _said(ASSISTANT, ""), idless)
    steps = atif.to_atif(run)["steps"]
    ids = [call["tool_call_id"] for step in steps for call in step["tool_calls"]]
    answered = [r["source_call_id"] for s in steps for r in s["observation"]["results"]]
    assert len(set(ids)) == len(ids) == 3
    assert answered == ids


def test_the_leaderboard_score_comes_out_in_front_and_only_numbers_come_at_all(caplog):
    """The midtier reads rewards.First(), so the aggregated score has to be first
    however the splits were ordered -- and rewards maps a name to a number, so a
    task returning a gold answer beside its score loses the answer, not the
    score."""
    results = [
        {
            "dictResult": {"score": 0.8, "gold": "Paris", "l": [1.0], "n": None},
            "type": "PUBLIC",
        },
        {"numericResult": {"value": 0.9}, "type": "AGGREGATED"},
        {"numericResult": {"value": 0.7}, "type": "PRIVATE"},
    ]
    with caplog.at_level(logging.WARNING, logger=atif.logger.name):
        rewards = atif.to_trial_result(_run(results=results))["verifier_result"][
            "rewards"
        ]
    assert list(rewards.items()) == [
        ("score", 0.9),
        ("public_score", 0.8),
        ("private_score", 0.7),
    ]
    assert all(key in caplog.text for key in ("gold", "l", "n"))


COSTS = {
    "inputTokensCostNanodollars": "1500000",
    "outputTokensCostNanodollars": "2500000",
}


@pytest.mark.parametrize(
    "metrics, expected",
    [
        # int64 proto fields serialize as strings, unlike the token counts.
        (COSTS, {"cost_usd": 0.004}),
        # Half a cost is an unknown total, not a partial one.
        (
            {"inputTokens": 100, "inputTokensCostNanodollars": "1"},
            {"n_input_tokens": 100},
        ),
        # A measured zero is data; only an unmeasured field becomes absence.
        (
            {"inputTokens": 0, "outputTokens": 5},
            {"n_input_tokens": 0, "n_output_tokens": 5},
        ),
        # Latency is the only per-turn timing kbench records, and harbor has no
        # field for it, so it is kept in extra rather than dropped.
        (
            {"inputTokens": 1200, "totalBackendLatencyMs": "987"},
            {"n_input_tokens": 1200},
        ),
    ],
)
def test_what_a_message_costs_and_what_is_left_unknown(metrics, expected):
    run = _run(conversations=[_chat(_said(ASSISTANT, "a"), metrics=metrics)])
    assert atif.to_trial_result(run)["agent_result"] == expected
    (step,) = atif.to_atif(run)["steps"]
    if "totalBackendLatencyMs" in metrics:
        assert step["metrics"]["extra"] == {"total_backend_latency_ms": 987}


TRACEBACK = "Traceback (most recent call last):\n  x\n"
CHAINED = (
    TRACEBACK + "KeyError: 'inner'\n\nDuring handling of the above exception, another "
    "exception occurred:\n\n" + TRACEBACK + "ValueError: outer\n"
)


@pytest.mark.parametrize(
    "error, expected, keeps_traceback",
    [
        # A message may contain the separator, or span lines.
        (
            TRACEBACK + "RuntimeError: failed: reason",
            ("RuntimeError", "failed: reason"),
            True,
        ),
        (TRACEBACK + "ValueError: a\nb", ("ValueError", "a\nb"), True),
        # Raised without arguments, so there is no separator at all.
        (TRACEBACK + "ValueError", ("ValueError", ""), True),
        # Anything not built in arrives qualified; harbor records __name__.
        (TRACEBACK + "mod.sub.Custom: mine", ("Custom", "mine"), True),
        # Harbor reads its own two timeout names as completed-without-reward,
        # but a kbench task that timed out has failed.
        (
            TRACEBACK + "AgentTimeoutError: slow",
            ("KbenchAgentTimeoutError", "slow"),
            True,
        ),
        # kbench only writes format_exc() today, but the proto field is free
        # text, so anything else degrades instead of guessing a type.
        ("kaboom", ("KbenchRunError", "kaboom"), False),
        # Truncated after the header, so there is no exception line to read.
        (TRACEBACK, ("KbenchRunError", TRACEBACK), False),
        # A chained traceback repeats the header, and the outermost exception is
        # the one that escaped, so the last block wins and not the first.
        (CHAINED, ("ValueError", "outer"), True),
    ],
)
def test_an_exception_is_read_back_out_of_the_traceback(
    error, expected, keeps_traceback
):
    info = atif.to_trial_result(_run(errorMessage=error))["exception_info"]
    assert (info["exception_type"], info["exception_message"]) == expected
    assert bool(info.get("exception_traceback")) == keeps_traceback
    # occurred_at is required, and now() would differ per conversion.
    no_end = atif.to_trial_result(_run(errorMessage=error, endTime=None))
    assert no_end["exception_info"]["occurred_at"] == "2026-01-01T00:00:00Z"


def test_rows_are_matched_by_id_never_by_position():
    """subruns[] is appended as rows finish, so under n_jobs > 1 its order is not
    the dataset's."""
    run = _run(subruns=[{"pyRunId": "row-Run #2"}, {"pyRunId": "row-Run #1"}])
    trajectory = atif.to_atif(run, subrun_paths={"row-Run #1": "first.atif.json"})
    (refs,) = trajectory["steps"][-1]["observation"]["results"]
    found = [r["trajectory_path"] for r in refs["subagent_trajectory_ref"]]
    assert found == ["first.atif.json"]
    assert _warnings(trajectory) == ["subrun_path_unknown"]
    # Aggregates written before the stubs existed have no subruns, so the count
    # is unavailable and the placeholder says less rather than 0.
    (step,) = atif.to_atif(_run(subruns=[]))["steps"]
    assert (
        step["message"]
        == "[placeholder] Aggregated from other runs. No conversation recorded."
    )


def test_a_field_the_converter_does_not_know_is_carried_through():
    """extra.kbench is whatever the trajectory has no field of its own for, so a
    newer kbench writing a field we have yet to map needs no code change."""
    kbench = atif.to_atif(_run(splitBreakdown={"public": 1}))["extra"]["kbench"]
    assert kbench["split_breakdown"] == {"public": 1}


# --- Writing the files, which is all the rest of kbench sees of this module ---


def _names(client, pattern="*") -> list[str]:
    return sorted(p.name for p in Path(client.directory).glob(pattern))


def test_a_run_writes_its_trajectory_beside_its_run_json(client, duck):
    """Three files per run, and the names differ only in suffix so a reader can
    pair them by eye."""
    _ran(duck, lambda llm: bool(llm.prompt("Hi")))
    assert _names(client) == [
        "T-run_id_Run_1_Duck.atif.json",
        "T-run_id_Run_1_Duck.result.json",
        "T-run_id_Run_1_Duck.run.json",
    ]
    written = json.loads(Path(client.directory, _names(client)[0]).read_text())
    assert written == atif.to_atif(_one(client))


def test_a_converter_that_raises_still_leaves_the_run(
    client, duck, monkeypatch, caplog
):
    """The run.json is the one file that must survive, so it is written first and
    a conversion failure only costs the other two."""

    def boom(*args, **kwargs):
        raise RuntimeError("converter broke")

    monkeypatch.setattr(atif, "to_atif", boom)
    monkeypatch.setattr(atif, "to_trial_result", boom)
    with caplog.at_level(logging.WARNING):
        _ran(duck, lambda llm: bool(llm.prompt("Hi")))
    assert _names(client) == ["T-run_id_Run_1_Duck.run.json"]
    # Silence would leave someone hunting for files that were never written.
    assert sum("converter broke" in r.message for r in caplog.records) == 2


def test_writing_can_be_turned_off(client, duck, monkeypatch):
    """Off is the whole switch: the run.json is unchanged, so nothing downstream
    of kbench notices."""
    monkeypatch.setattr(config, "write_atif", False)
    _ran(duck, lambda llm: bool(llm.prompt("Hi")))
    assert _names(client) == ["T-run_id_Run_1_Duck.run.json"]


def test_an_eval_writes_three_files_per_row_and_the_parent_names_them(client, duck):
    """The parent's refs are the only link from an eval to its rows, and they are
    filenames the caller resolved -- so every one must be a file on disk."""
    _dataset_eval(duck)
    assert len(_names(client)) == 4 * 3
    parent = json.loads(
        Path(client.directory, "whole_ds-run_id_Run_1_Duck.atif.json").read_text()
    )
    (results,) = parent["steps"][-1]["observation"]["results"]
    refs = results["subagent_trajectory_ref"]
    assert [ref["extra"]["kbench_py_run_id"] for ref in refs] == [
        "row_qa-Run #1",
        "row_qa-Run #2",
        "row_qa-Run #3",
    ]
    assert all(Path(client.directory, ref["trajectory_path"]).exists() for ref in refs)
    assert _warnings(parent) == []
    # Both sides of the link agree on the group, which is how harbor collects a
    # job's trials into one dataset.
    sources = {
        json.loads(p.read_text())["source"]
        for p in Path(client.directory).glob("*.result.json")
    }
    assert sources == {"whole_ds"}


def test_removing_the_row_files_leaves_no_orphans(client, duck):
    """remove_run_files is what the documented eval example passes. Unfixed it
    would leave 2N files whose run.json is gone."""
    _dataset_eval(duck, remove_run_files=True)
    assert _names(client) == [
        "whole_ds-run_id_Run_1_Duck.atif.json",
        "whole_ds-run_id_Run_1_Duck.result.json",
        "whole_ds-run_id_Run_1_Duck.run.json",
    ]
    parent = json.loads(
        Path(client.directory, "whole_ds-run_id_Run_1_Duck.atif.json").read_text()
    )
    # Named nothing rather than named the deleted files, and the rows are still
    # summarised, so the outcomes survive their transcripts.
    assert "observation" not in parent["steps"][-1]
    assert _warnings(parent) == ["subrun_path_unknown"] * 3
    assert len(parent["extra"]["kbench"]["subruns"]) == 3


def test_an_aggregate_gets_a_trajectory_too(client, duck):
    """merge_results_from_runfiles writes its own file and never reaches
    store_run, so without this one path would have no trajectory at all."""

    def rep(llm) -> float:
        llm.prompt("Hi")
        return 0.5

    _ran(duck, rep, "Rep", times=2)
    inputs = sorted(str(p) for p in Path(client.directory).glob("Rep*.run.json"))
    output = serialization.merge_results_from_runfiles(
        inputs,
        lambda rs: sum(r["numericResult"].get("value", 0.0) for r in rs) / len(rs),
        output_directory=str(client.directory),
        delete_run_files=True,
    )
    # The merged runs took their own trajectories with them.
    assert _names(client) == [
        "Rep-Run_aggregated.atif.json",
        "Rep-Run_aggregated.result.json",
        "Rep-Run_aggregated.run.json",
    ]
    trajectory = json.loads(Path(output.replace(".run.", ".atif.")).read_text())
    assert trajectory["final_metrics"]["extra"]["kbench_result"] == {"score": 0.5}


def test_choose_keeps_or_drops_a_runs_three_files_together(client, duck, monkeypatch):
    """%choose picks one run out of a notebook's leftovers. All three files
    describe that run, so keeping the run.json alone would leave a trajectory of
    a run the user just discarded."""
    _ran(duck, lambda llm: bool(llm.prompt("Hi")), "T", times=2)
    monkeypatch.setattr(ipython_magics, "WORKING_DIR", Path(client.directory))
    monkeypatch.setattr(
        ipython_magics.core.getipython,
        "get_ipython",
        lambda: SimpleNamespace(user_ns={}),
    )
    ipython_magics.choose("T")
    # The newest run, whole. Run #1's files went with its run.json.
    assert _names(client) == [
        "T-run_id_Run_2_Duck.atif.json",
        "T-run_id_Run_2_Duck.result.json",
        "T-run_id_Run_2_Duck.run.json",
    ]
