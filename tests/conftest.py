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

from kaggle_benchmarks import clients, config, contexts, events


@pytest.fixture(autouse=True)
def context(monkeypatch):
    with contexts.enter():
        config.interactive_mode = False
        events.manager.listeners = []
        config.ui_handler = None
        monkeypatch.setattr("kaggle_benchmarks.client", clients.InMemoryClient())
        yield
