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

from kaggle_benchmarks import (
    actors,
    assertions,
    chats,
    clients,
    content_types,
    envs,
    kaggle,
    orchestration,
    prompting,
    rooms,
    tasks,
    tools,
    ui,
    utils,
)
from kaggle_benchmarks._config import ExecutionMode, config
from kaggle_benchmarks.actors import Actor, LLMChat, system, user
from kaggle_benchmarks.chats import last_reasoning_traces
from kaggle_benchmarks.kaggle.model_proxy import raise_for_missing_model_proxy_config
from kaggle_benchmarks.rooms import ChatRoom, Participant
from kaggle_benchmarks.runs import Run, Runs
from kaggle_benchmarks.tasks import benchmark, task
from kaggle_benchmarks.usage import Usage


class _NotConfiguredLLM(LLMChat):
    """Stand-in for an LLMChat when no model provider is configured.

    Allows ``import kaggle_benchmarks`` to succeed without credentials,
    but raises a clear error with setup instructions when the model is
    actually used (via ``prompt()``, ``respond()``, etc.).
    """

    def __init__(self, name: str = "not-configured"):
        super().__init__(name=name)

    def invoke(self, *args, **kwargs):
        raise_for_missing_model_proxy_config()


if kaggle.is_configured():
    llm = kaggle.load_default_model()
    judge_llm = kaggle.load_judge_model()
    llms = kaggle.load_available_models()
else:
    llm = _NotConfiguredLLM("llm")
    judge_llm = _NotConfiguredLLM("judge_llm")
    llms = {}


client: clients.Client = clients.resolve_client()
config.apply()


__version__ = "0.6.0"
