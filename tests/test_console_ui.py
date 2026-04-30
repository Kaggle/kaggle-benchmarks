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

import io

import pytest

from kaggle_benchmarks import (
    actors,
    assertions,
    chats,
    config,
    events,
    system,
    tasks,
    ui,
    user,
)
from kaggle_benchmarks.actors import llms
from tests.mocks import MockedChat


class _StreamingLLM(actors.LLMChat):
    """LLM whose invoke() returns an Iterator (exercises the streaming path)."""

    def __init__(self, chunks, name="Stream", **kwargs):
        super().__init__(name=name, **kwargs)
        self._chunks = list(chunks)

    def invoke(self, messages, **kwargs):
        return iter(self._chunks)


class _NonStreamingLLM(actors.LLMChat):
    """LLM whose invoke() returns an LLMResponse (non-streaming path)."""

    def __init__(self, content, name="NonStream", **kwargs):
        super().__init__(name=name, **kwargs)
        self._content = content

    def invoke(self, messages, **kwargs):
        return llms.LLMResponse(content=self._content)


@pytest.fixture
def duck():
    yield MockedChat.from_contents(["quack"], name="Duck", cycle=True)


def _make_handler(quiet=False, color=False):
    output = io.StringIO()
    handler = ui.console.ConsoleUI(quiet=quiet, color=color, output=output)
    events.manager.bind(handler)
    return handler, output


def _get_output(output):
    return output.getvalue()


class TestQuietMode:
    def test_quiet_mode_matches_original_behavior(self):
        handler, output = _make_handler(quiet=True)
        with chats.new("test") as chat:
            user.send("hi")
            system.send("The end")

        captured = _get_output(output)
        assert captured.strip() == str(chat).strip()

    def test_quiet_mode_streaming(self):
        handler, output = _make_handler(quiet=True)
        with chats.new("test"):
            user.stream("one two three".split())

        captured = _get_output(output)
        assert "onetwothree" in captured

    def test_quiet_mode_ignores_runs(self):
        handler, output = _make_handler(quiet=True)

        @tasks.task(name="Quiet Task", store_task=False, store_run=False)
        def quiet_task():
            user.send("hello")

        quiet_task.run()
        captured = _get_output(output)
        # In quiet mode, run headers should not appear
        assert "TASK" not in captured
        assert "====" not in captured


class TestRichMode:
    def test_run_header_and_footer(self):
        handler, output = _make_handler()

        @tasks.task(name="My Task", store_task=False, store_run=False)
        def my_task():
            pass

        my_task.run()
        captured = _get_output(output)
        assert "=" * 80 in captured
        assert "TASK: My Task:" in captured
        assert "RESULT:" in captured

    def test_prompt_and_response(self, duck):
        handler, output = _make_handler()

        @tasks.task(name="Chat Task", store_task=False, store_run=False)
        def chat_task(llm):
            llm.prompt("What sound do you make?")

        chat_task.run(duck)
        captured = _get_output(output)
        assert "[PROMPT]" in captured
        assert "What sound do you make?" in captured
        assert "[RESPONSE: Duck]" in captured
        assert "quack" in captured

    def test_system_message(self):
        handler, output = _make_handler()

        @tasks.task(name="Sys Task", store_task=False, store_run=False)
        def sys_task():
            system.send("System info here")

        sys_task.run()
        captured = _get_output(output)
        assert "[SYSTEM]" in captured
        assert "System info here" in captured

    def test_assertion_table_pass(self):
        handler, output = _make_handler()

        @tasks.task(name="Assert Task", store_task=False, store_run=False)
        def assert_task():
            assertions.assert_true(True, expectation="Value should be true")

        assert_task.run()
        captured = _get_output(output)
        assert "✅ Pass" in captured
        assert "Value should be true" in captured
        assert "Expectation" in captured

    def test_assertion_table_fail(self, monkeypatch):
        monkeypatch.setattr(config, "continue_with_exceptions", True)
        handler, output = _make_handler()

        @tasks.task(name="Fail Task", store_task=False, store_run=False)
        def fail_task():
            assertions.assert_true(False, expectation="This should fail")

        fail_task.run()
        captured = _get_output(output)
        assert "❌ Fail" in captured
        assert "This should fail" in captured

    def test_assertion_table_mixed(self, monkeypatch):
        monkeypatch.setattr(config, "continue_with_exceptions", True)
        handler, output = _make_handler()

        @tasks.task(name="Mixed Task", store_task=False, store_run=False)
        def mixed_task():
            assertions.assert_true(True, expectation="First check")
            assertions.assert_true(False, expectation="Second check")
            assertions.assert_equal("a", "a", expectation="Third check")

        mixed_task.run()
        captured = _get_output(output)
        assert captured.count("✅ Pass") == 2
        assert captured.count("❌ Fail") == 1
        assert "RESULT:" in captured

    def test_error_run(self, monkeypatch):
        monkeypatch.setattr(config, "continue_with_exceptions", True)
        handler, output = _make_handler()

        @tasks.task(name="Error Task", store_task=False, store_run=False)
        def error_task():
            raise ValueError("Something broke")

        error_task.run()
        captured = _get_output(output)
        assert "ERROR:" in captured
        assert "Something broke" in captured

    def test_result_format(self, duck):
        handler, output = _make_handler()

        @tasks.task(name="Bool Task", store_task=False, store_run=False)
        def bool_task(llm) -> bool:
            return llm.prompt("test") == "quack"

        bool_task.run(duck)
        captured = _get_output(output)
        assert "RESULT:" in captured

    def test_streaming(self):
        handler, output = _make_handler()

        @tasks.task(name="Stream Task", store_task=False, store_run=False)
        def stream_task():
            user.stream(["hello", " ", "world"])

        stream_task.run()
        captured = _get_output(output)
        assert "hello world" in captured


class TestColorMode:
    def test_color_pass(self):
        handler, output = _make_handler(color=True)

        @tasks.task(name="Color Task", store_task=False, store_run=False)
        def color_task():
            assertions.assert_true(True, expectation="Check it")

        color_task.run()
        captured = _get_output(output)
        # ANSI green should wrap the PASS icon
        assert "\033[32m✅ Pass\033[0m" in captured

    def test_color_fail(self, monkeypatch):
        monkeypatch.setattr(config, "continue_with_exceptions", True)
        handler, output = _make_handler(color=True)

        @tasks.task(name="Color Fail", store_task=False, store_run=False)
        def color_fail_task():
            assertions.assert_true(False, expectation="Check it")

        color_fail_task.run()
        captured = _get_output(output)
        # ANSI red should wrap the FAIL icon
        assert "\033[31m❌ Fail\033[0m" in captured

    def test_color_headers(self):
        handler, output = _make_handler(color=True)

        @tasks.task(name="Color Header", store_task=False, store_run=False)
        def color_header_task():
            user.send("hello")

        color_header_task.run()
        captured = _get_output(output)
        # ANSI bold should wrap the title bar
        assert "\033[1m" in captured
        # ANSI cyan should wrap [PROMPT]
        assert "\033[36m" in captured

    def test_no_color_has_no_ansi(self):
        handler, output = _make_handler(color=False)

        @tasks.task(name="No Color", store_task=False, store_run=False)
        def no_color_task():
            assertions.assert_true(True, expectation="Check it")
            user.send("hello")

        no_color_task.run()
        captured = _get_output(output)
        assert "\033[" not in captured


class TestNestedRuns:
    def test_subrun_formatting(self, duck):
        handler, output = _make_handler()

        @tasks.task(name="Inner Task", store_task=False, store_run=False)
        def inner_task(llm):
            llm.prompt("inner question")

        @tasks.task(name="Outer Task", store_task=False, store_run=False)
        def outer_task(llm):
            inner_task.run(llm)

        outer_task.run(duck)
        captured = _get_output(output)
        # Both tasks should appear
        assert "Outer Task" in captured
        assert "Inner Task" in captured
        assert "SUBTASK" in captured


class TestLLMResponseRendering:
    """Verify the rendering ordering for each LLMChat.respond() invoke branch.

    These exercise the actual code paths in actors/llms.py — MockedChat only
    hits the LLMMessage branch, so we need fakes for LLMResponse and Iterator.
    """

    def test_non_streaming_quiet_preserves_newline(self):
        """LLMResponse path in quiet mode must terminate each message with \\n.

        Regression: an earlier version swapped chat.append/status order so
        new_message fired with status=RUNNING, causing _quiet_new_message to
        omit the trailing newline and glue messages together.
        """
        handler, output = _make_handler(quiet=True)
        llm = _NonStreamingLLM(content="alpha")
        with chats.new("test"):
            llm.prompt("first")
            llm.prompt("second")
        captured = _get_output(output)
        # Each message must end with a newline; lines must not be glued.
        assert "alpha\n" in captured
        assert "alpha  " not in captured  # not followed by next message inline

    def test_streaming_header_before_chunks_in_rich_mode(self):
        """Streaming path must dispatch new_message BEFORE chunks stream in,
        so the rich UI's [RESPONSE: name] header appears above the tokens."""
        handler, output = _make_handler()

        @tasks.task(name="StreamTask", store_task=False, store_run=False)
        def stream_task(llm):
            llm.prompt("go")

        stream_task.run(_StreamingLLM(chunks=["hel", "lo ", "world"]))
        captured = _get_output(output)
        header_idx = captured.find("[RESPONSE: Stream]")
        body_idx = captured.find("hello world")
        assert header_idx != -1, captured
        assert body_idx != -1, captured
        assert header_idx < body_idx, (
            f"[RESPONSE] header at {header_idx} must precede streamed body "
            f"at {body_idx}:\n{captured}"
        )

    def test_streaming_no_duplicate_body(self):
        """Streamed text should appear exactly once, not also re-printed by
        a later new_message dispatch."""
        handler, output = _make_handler()

        @tasks.task(name="StreamDup", store_task=False, store_run=False)
        def stream_task(llm):
            llm.prompt("go")

        stream_task.run(_StreamingLLM(chunks=["uniq", "ue!"]))
        captured = _get_output(output)
        assert captured.count("unique!") == 1, captured

    def test_non_streaming_rich_renders_response(self):
        """LLMResponse path inside @task should print header + body once."""
        handler, output = _make_handler()

        @tasks.task(name="NonStreamTask", store_task=False, store_run=False)
        def t(llm):
            llm.prompt("go")

        t.run(_NonStreamingLLM(content="answer-text"))
        captured = _get_output(output)
        assert "[RESPONSE: NonStream]" in captured
        assert captured.count("answer-text") == 1, captured

    def test_invisible_to_llm_messages_still_render(self):
        """Messages with is_visible_to_llm=False (e.g. tool debug output)
        should still appear in the console — that flag controls inclusion
        in the LLM context window, not user-facing visibility."""
        from kaggle_benchmarks import actors, messages

        handler, output = _make_handler(quiet=False)

        @tasks.task(name="ToolTask", store_task=False, store_run=False)
        def tool_task():
            tool_actor = actors.Actor(name="Docker", role="tool")
            chats.get_current_chat().append(
                messages.Message(
                    sender=tool_actor,
                    content="container exited with code 0",
                    is_visible_to_llm=False,
                )
            )

        tool_task.run()
        captured = _get_output(output)
        assert "[TOOL: Docker]" in captured, captured
        assert "container exited with code 0" in captured, captured

    def test_streaming_does_not_double_print(self):
        """If a message is streamed and *then* appended (the inverse of the
        canonical order), new_message should still print the role header but
        skip the body since chunks already rendered it."""
        from kaggle_benchmarks import actors, messages, utils

        handler, output = _make_handler(quiet=False)

        @tasks.task(name="Stream Test", store_task=False, store_run=False)
        def stream_task():
            msg = messages.Message(
                sender=actors.system, content="", _status=utils.Status.RUNNING
            )
            msg.stream(iter(["hello", " world"]))
            chats.get_current_chat().append(msg)

        stream_task.run()
        captured = _get_output(output)
        assert captured.count("hello world") == 1, captured
        # Header should still appear so the [SYSTEM] label is visible.
        assert "[SYSTEM]" in captured

    def test_streaming_with_usage_prints_metrics_once(self):
        """For assistant-role streaming responses with usage, METRICS must
        be printed exactly once (by end_run from chat.usage), not also by
        end_content per-message."""
        from dataclasses import dataclass, field

        from kaggle_benchmarks import actors, messages, utils

        @dataclass
        class Chunk:
            content: str
            meta: dict = field(default_factory=dict)

        handler, output = _make_handler()

        @tasks.task(name="Metrics Task", store_task=False, store_run=False)
        def metrics_task():
            llm_actor = actors.Actor(name="TestLLM", role="assistant")
            msg = messages.Message(
                sender=llm_actor, content="", _status=utils.Status.RUNNING
            )
            chunks = [
                Chunk("hello"),
                Chunk(" world", meta={"input_tokens": 10, "output_tokens": 5}),
            ]
            chats.get_current_chat().append(msg)
            msg.stream(chunks)
            msg.status = utils.Status.SUCCESS

        metrics_task.run()
        captured = _get_output(output)
        assert captured.count("METRICS:") == 1, captured

    def test_assertion_table_narrow_terminal(self):
        """expect_width must stay positive even when terminal is very narrow
        and run depth is deep, otherwise textwrap.wrap raises ValueError."""
        output = io.StringIO()
        handler = ui.console.ConsoleUI(
            quiet=False, color=False, output=output, width=30, min_width=30
        )
        handler._run_depth = 8  # raw expect_width would be -4 without the floor
        results = [
            assertions.AssertionResult(passed=True, expectation="Some expectation")
        ]
        table = handler._format_assertion_table(results)
        assert isinstance(table, str)
        assert "Some expectation" in table or "Some" in table

    def test_switching_ui_unbinds_old_handler(self, cfg):
        """Switching UI mode must unbind the previous handler so events
        aren't dispatched to both."""
        cfg.enable_interactive_mode()  # binds PanelUI
        old_handler = cfg.ui_handler
        from kaggle_benchmarks.ui import panel as panel_ui

        assert isinstance(old_handler, panel_ui.PanelUI)
        assert old_handler in events.manager.listeners

        cfg.enable_console_mode()  # should unbind PanelUI, bind ConsoleUI
        assert old_handler not in events.manager.listeners
        assert isinstance(cfg.ui_handler, ui.console.ConsoleUI)
        assert cfg.ui_handler in events.manager.listeners

        # And the reverse: console -> panel must also unbind.
        console_handler = cfg.ui_handler
        cfg.enable_interactive_mode()
        assert console_handler not in events.manager.listeners
        assert isinstance(cfg.ui_handler, panel_ui.PanelUI)

    def test_llm_message_branch_dispatches_new_message_once(self):
        """LLMMessage returned from invoke() should produce exactly one
        new_message event for the response (the chat.append in respond()).

        Guards against future regressions if invoke() starts using
        LLMMessage.from_chunks (which dispatches new_message itself) — that
        would need chat.history.append, not chat.append, to avoid a duplicate.
        """
        events_seen = []

        class Spy:
            def new_message(self, chat, message):
                events_seen.append(message)

        events.manager.listeners = []
        config.ui_handler = None
        events.manager.bind(Spy())

        llm = MockedChat.from_contents(["only-once"])
        with chats.new("t"):
            user.send("hi")
            llm.respond()

        # Filter to just message events (not chat-open events).
        msgs = [m for m in events_seen if not isinstance(m, chats.Chat)]
        # Expect exactly two: the user prompt, and the LLM response.
        # If respond() dispatches twice for the LLMMessage branch this jumps to 3.
        assert len(msgs) == 2, (
            f"Expected 2 message events, got {len(msgs)}: "
            f"{[m.content for m in msgs]}"
        )
        assert msgs[-1].content == "only-once"
