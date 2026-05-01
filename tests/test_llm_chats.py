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

import json

import pytest

from kaggle_benchmarks import actors, chats, contexts, events, prompting, utils
from kaggle_benchmarks.actors.llms import LLMResponse
from kaggle_benchmarks.content_types import images, videos
from kaggle_benchmarks.llm_messages import LLMMessage
from kaggle_benchmarks.prompting import handler
from tests.mocks import MockedChat


class Ferret(actors.LLMChat):
    def __init__(self):
        super().__init__(name="Ferret")
        self.stream_responses = False

    def invoke(self, messages, system=None, **kwargs):
        if not self.stream_responses:
            return LLMResponse(
                content=json.dumps(
                    dict(
                        messages=[[m.sender.name.lower(), m.content] for m in messages],
                        system=system,
                    )
                )
            )

        def stream_generator():
            yield LLMResponse(content="stream", meta={"input_tokens": 10})
            yield LLMResponse(
                content="ing", meta={"input_tokens": 10, "output_tokens": 1}
            )
            yield LLMResponse(
                content="...", meta={"input_tokens": 10, "output_tokens": 2}
            )

        return stream_generator()


def test_prompt_without_context():
    llm = Ferret()
    r = llm.prompt("A")
    assert {"messages": [["user", "A"]], "system": None} == json.loads(r)


def test_respond():
    llm = Ferret()

    with chats.new("Test") as t:
        actors.user.send("A")
        assert len(t.messages) == 1

        r = llm.respond()
        assert len(t.messages) == 2
        assert {"messages": [["user", "A"]], "system": None} == json.loads(r.text)


def test_chat_context():
    llm = Ferret()
    llm.prompt("<should not be visible in the context>")

    with chats.new(system_instructions="S") as t:
        assert t.status == utils.Status.RUNNING

        r = llm.prompt("A")
        assert {
            "messages": [["system", "S"], ["user", "A"]],
            "system": None,
        } == json.loads(r)

        r = llm.prompt("B")
        response = json.loads(r)

        assert response["system"] is None
        assert 4 == len(response["messages"])
        assert ["system", "S"] == response["messages"][0]
        assert ["user", "A"] == response["messages"][1]
        assert llm.name.lower() == response["messages"][2][0]
        assert ["user", "B"] == response["messages"][3]

    assert t.status == utils.Status.SUCCESS


def test_structured():
    llm = Ferret()

    class F:
        pass

    value = F()

    @handler(types=F)
    def _(cls):
        yield ""
        return value

    response = llm.prompt("Test", schema=F)
    assert isinstance(response, F)
    assert value is response

    @handler(types=F)
    def _(cls):
        value = yield ""
        raise prompting.ResponseParsingError(
            error="Bad response", schema=cls, value=value
        )

    with chats.new() as t:
        with pytest.raises(prompting.ResponseParsingError):
            llm.prompt("test_value", schema=F)
        assert "Bad response" in t.messages[-1].text
        assert "test_value" in t.messages[-1].text
        assert "F" in t.messages[-1].text

    @handler(types=F)
    def _(cls):
        yield ""
        yield "nonsense"
        return F()

    with pytest.raises(prompting.SchemaError):
        llm.prompt("Test", schema=F)


def test_streaming_prompt():
    llm = Ferret()
    # Explicitly set stream mode.
    llm.stream_responses = True

    with chats.new("Test Streaming") as t:
        response_content = llm.prompt("stream this")
        assert response_content == "streaming..."

        # The last message in the chat is the one from the LLM.
        last_message = t.messages[-1]
        assert last_message.content == "streaming..."
        assert last_message.sender is llm
        assert last_message._meta["input_tokens"] == 10
        assert last_message._meta["output_tokens"] == 2


def test_nested_chat_id():
    llm = Ferret()
    with chats.new("root") as root:
        sub = chats.Chat(name="sub")
        chats.get_current_chat().append(sub)
        with contexts.enter(chat=sub):
            llm.prompt("Hi")

        sub.name += " - analysis"

    assert root.history[0] is sub
    assert sub.id.startswith("sub - analysis-")
    assert len(sub.history) == 2


def test_chat_usage_aggregation():
    """Test that chat usage properties aggregate token usage from all assistant messages."""
    llm = Ferret()
    llm.stream_responses = True

    with chats.new("Test Usage") as t:
        llm.prompt("first")
        llm.prompt("second")

        # Each streaming response yields: input_tokens=10, output_tokens=2
        # Two prompts = 2 * 10 = 20 input tokens, 2 * 2 = 4 output tokens
        assert t.usage.input_tokens == 20
        assert t.usage.output_tokens == 4


def test_chat_usage_empty():
    """Test that chat usage properties return zero/None for empty chat."""
    with chats.new("Empty") as t:
        assert t.usage.input_tokens is None
        assert t.usage.output_tokens is None
        assert t.usage.input_tokens_cost_nanodollars is None
        assert t.usage.output_tokens_cost_nanodollars is None
        assert t.usage.total_backend_latency_ms is None


def test_video_message_payload():
    """Test that a VideoURL message produces the correct payload for the OpenAI backend."""
    video = videos.from_url("https://www.youtube.com/watch?v=abc123")

    from kaggle_benchmarks import messages

    msg = messages.Message(sender=actors.user, content=video)
    assert msg.payload == [
        {
            "type": "image_url",
            "image_url": {"url": "https://www.youtube.com/watch?v=abc123"},
        }
    ]


def test_prompt_with_image_and_video():
    """Test that prompt() with both image and video sends them as separate messages."""
    red_pixel_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    img = images.from_base64(red_pixel_b64, format="png")
    video = videos.from_url("https://www.youtube.com/watch?v=abc123")

    with chats.new("image_and_video") as t:
        # Manually send image and video, then prompt, to verify message ordering.
        # We don't call llm.prompt() directly because Ferret's invoke() can't
        # serialize image/video content to JSON.
        actors.user.send(img)
        actors.user.send(video)
        actors.user.send("Describe both")

        assert len(t.messages) == 3
        assert isinstance(t.messages[0].content, images.ImageBase64)
        assert isinstance(t.messages[1].content, videos.VideoURL)
        assert t.messages[2].content == "Describe both"


def test_chat_fork():
    with chats.new("Parent") as p:
        actors.user.send("Hello")

        with chats.fork("Forked") as f:
            assert f.name == "Forked"
            assert len(f.history) == 1
            assert f.history[0].content == "Hello"
            actors.user.send("World")
            assert len(f.history) == 2

        assert len(p.history) == 2
        assert p.history[1] is f

        with chats.fork("Orphaned", orphan=True) as of:
            assert of.name == "Orphaned"
            assert len(of.history) == 1  # Only messages are copied to the fork
            actors.user.send("Bye")
            assert len(of.history) == 2

        assert len(p.history) == 2


def test_invoke_llmmessage():
    mocked_chat = MockedChat.from_contents(["test response"])
    messages = [LLMMessage(sender=actors.user, content="hello")]

    response = mocked_chat.invoke(messages=messages, temperature=0.5)

    assert isinstance(response, LLMMessage)
    assert response.content == "test response"
    assert response.sender is mocked_chat
    assert len(mocked_chat.invocations) == 1
    assert mocked_chat.invocations[0][0] == messages
    assert mocked_chat.invocations[0][1] == {"temperature": 0.5}


def test_panel_ui_tolerates_message_update_before_new_message():
    """PanelUI.message_update must not crash on messages it hasn't
    registered via new_message yet.

    We call message_update directly rather than going through
    llm.prompt() because contexts.enter() swallows exceptions when
    there is no parent run, which would mask the KeyError."""
    from kaggle_benchmarks.messages import Message
    from kaggle_benchmarks.ui import panel as panel_ui

    handler = panel_ui.PanelUI()
    msg = Message(content="test", sender=actors.user, _status=utils.Status.RUNNING)

    # Must not raise KeyError for messages not yet seen via new_message.
    handler.message_update(msg, utils.Status.SUCCESS)


def test_panel_ui_tolerates_unregistered_keys():
    """Companion to test_panel_ui_tolerates_message_update_before_new_message
    for the other handlers. Under n_jobs > 1, joblib's threading backend
    dispatches lifecycle events from worker threads through the global
    events.manager, and PanelUI.new_chat can hit its `elif parent is None`
    branch and skip registering the chat — leaving downstream handlers to
    fire on keys they never saw. Each must tolerate that."""
    from unittest.mock import Mock

    from kaggle_benchmarks.messages import Message
    from kaggle_benchmarks.ui import panel as panel_ui

    handler = panel_ui.PanelUI()
    msg = Message(content="x", sender=actors.user, _status=utils.Status.RUNNING)
    run = Mock()

    handler.new_chunk(msg, "chunk")
    handler.end_content(msg)
    handler.new_tool_call(msg, Mock())

    starting_depth = handler.depth
    handler.end_run(run)
    assert handler.depth == starting_depth - 1, (
        "end_run must still decrement depth even when run is unregistered, "
        "to stay balanced with new_run."
    )


def test_panel_ui_new_run_tolerates_unregistered_parent():
    """Deterministic stand-in for the n_jobs > 1 race: when self.depth
    is concurrently inflated by another thread, new_run takes the
    else-branch and looks up self[run.parent] — which may not be
    registered. Must not KeyError."""
    from kaggle_benchmarks import results, runs, tasks
    from kaggle_benchmarks.ui import panel as panel_ui

    handler = panel_ui.PanelUI()

    # Simulate depth corruption from concurrent threads: set depth >= 1
    # so new_run takes the parent-lookup branch instead of add_card.
    # run.parent is unregistered, so this must not KeyError.
    handler.depth = 2
    dummy_task = tasks.Task(
        func=lambda: None, name="dummy-new-run", store_task=False, store_run=False
    )
    unregistered_parent_run = runs.Run(task=dummy_task, result=results.PENDING)
    handler.new_run(unregistered_parent_run)


def test_panel_ui_concurrent_prompts():
    """PanelUI must not crash when multiple threads dispatch events
    concurrently, as happens with evaluate(n_jobs > 1)."""
    import concurrent.futures

    from kaggle_benchmarks.ui import panel as panel_ui

    handler = panel_ui.PanelUI()

    # Stub new_chunk to avoid Panel's .stream() rejecting LLMResponse.
    def safe_new_chunk(message, chunk):
        if message in handler:
            pass

    handler.new_chunk = safe_new_chunk

    events.manager.bind(handler)
    errors = []

    def run_prompt(i):
        try:
            with contexts.enter():
                llm = Ferret()
                with chats.new(f"thread-{i}"):
                    llm.prompt(f"hello from thread {i}")
        except Exception as e:
            errors.append(e)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(run_prompt, i) for i in range(8)]
            concurrent.futures.wait(futures)
    finally:
        events.manager.unbind(handler)

    assert not errors, f"Concurrent prompts raised: {errors}"


def test_panel_ui_streaming_with_bound_handler():
    """The streaming path must append the message (registering it in
    PanelUI.shadows via new_message) before calling response.stream()
    (which dispatches new_chunk). This test locks in that ordering."""
    from kaggle_benchmarks.ui import panel as panel_ui

    handler = panel_ui.PanelUI()

    # Replace new_chunk with a stub that verifies the message is
    # registered (the ordering invariant) without calling Panel's
    # internal .stream(), which rejects mock LLMResponse objects.
    def ordering_check(message, chunk):
        assert message in handler

    handler.new_chunk = ordering_check

    events.manager.bind(handler)
    try:
        llm = Ferret()
        llm.stream_responses = True
        with chats.new("test"):
            llm.prompt("hello")
    finally:
        events.manager.unbind(handler)


def test_panel_ui_depth_is_per_thread():
    """depth must be per-thread so concurrent new_run/end_run
    pairs don't corrupt each other's nesting."""
    import concurrent.futures
    import time

    from kaggle_benchmarks.ui import panel as panel_ui

    handler = panel_ui.PanelUI()

    errors = []

    def run_nested(i):
        try:
            handler.depth += 1
            assert handler.depth == 1, (
                f"thread-{i}: depth should be 1 after increment, got {handler.depth}"
            )
            time.sleep(0.01)  # force interleaving
            handler.depth -= 1
            assert handler.depth == 0, (
                f"thread-{i}: depth should be 0 after decrement, got {handler.depth}"
            )
        except AssertionError as e:
            errors.append(e)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run_nested, i) for i in range(8)]
        concurrent.futures.wait(futures)

    assert not errors, f"Depth leaked across threads: {errors}"


def test_event_manager_dispatch_tolerates_unbind():
    """dispatch must not crash if a listener unbinds itself during dispatch."""
    from kaggle_benchmarks.events import EventManager

    manager = EventManager()

    class SelfUnbinder:
        def on_event(self):
            manager.unbind(self)

    class Counter:
        count = 0

        def on_event(self):
            self.count += 1

    unbinder = SelfUnbinder()
    counter = Counter()
    manager.bind(unbinder)
    manager.bind(counter)

    # Must not raise RuntimeError from list mutation during iteration
    manager.dispatch("on_event")
    assert counter.count == 1
