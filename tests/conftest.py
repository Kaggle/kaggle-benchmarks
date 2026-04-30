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

from kaggle_benchmarks import ExecutionMode, clients, config, contexts, events
from tests.mocks import MockedChat


@pytest.fixture(autouse=True)
def context(monkeypatch):
    with contexts.enter():
        config.execution_mode = ExecutionMode.TESTING
        config.interactive_mode = False
        config.console_mode = False
        config.console_quiet = False
        config.console_color = None
        events.manager.listeners = []
        config.ui_handler = None
        config.apply()
        monkeypatch.setattr("kaggle_benchmarks.client", clients.InMemoryClient())
        yield


@pytest.fixture()
def cfg():
    before = config.__dict__.copy()
    yield config
    config.__dict__.update(before)
    config.apply()


@pytest.fixture()
def duck():
    yield MockedChat.from_contents(["quack"], name="Duck", cycle=True)


@pytest.fixture()
def goose():
    yield MockedChat.from_contents(["honk"], name="Goose", cycle=True)
