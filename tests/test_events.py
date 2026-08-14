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

import pytest

from kaggle_benchmarks import chats, events, system, ui, user


class EventLogger:
    def __init__(self):
        self.events = {}

    def __getattr__(self, name):
        return lambda *x: self.events.setdefault(name, []).append(x)


@pytest.fixture
def logger():
    logger = EventLogger()
    events.manager.bind(logger)
    yield logger
    events.manager.unbind(logger)


def test_start_end_chat(logger):
    with chats.new("test"):
        assert len(logger.events["new_chat"]) == 1
        assert len(logger.events["new_event"]) == 1
        user.send("hi")
        assert len(logger.events["new_event"]) == 2

    assert len(logger.events["end_chat"]) == 1


def test_new_event(logger):
    user.send("hi")
    assert len(logger.events["new_event"]) == 1

    with chats.new("test"):
        user.send("hi")
        assert len(logger.events["new_event"]) == 3


def test_console_ui(capsys):
    handler = ui.console.ConsoleUI(tab_size=2)
    events.manager.bind(handler)
    with chats.new("test") as chat:
        user.send("hi")
        user.stream("one two three".split())
        with chats.new("inner"):
            user.send("inner message")
        system.send("The end")

    captured = capsys.readouterr()
    assert captured.out.strip() == str(chat).strip()


def test_panel_ui():
    from kaggle_benchmarks.ui import panel

    handler = panel.PanelUI()
    events.manager.bind(handler)
    with chats.new("test") as chat:
        assert chat in handler
        msg = user.send("hi")
        assert msg in handler
