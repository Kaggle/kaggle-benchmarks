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

from kaggle_benchmarks import assertions, chats, config, events, system, tasks, ui, user
from tests.mocks import MockedChat


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
