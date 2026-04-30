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
    tasks,
    tools,
    ui,
    utils,
)
from kaggle_benchmarks._config import ExecutionMode, config
from kaggle_benchmarks.actors import Actor, LLMChat, system, user
from kaggle_benchmarks.chats import last_reasoning_traces
from kaggle_benchmarks.kaggle.model_proxy import validate_model_proxy_config
from kaggle_benchmarks.runs import Run, Runs
from kaggle_benchmarks.tasks import benchmark, task
from kaggle_benchmarks.usage import Usage

if kaggle.is_configured():
    llm = kaggle.load_default_model()
    judge_llm = kaggle.load_judge_model()
    llms = kaggle.load_available_models()
else:
    validate_model_proxy_config()


client: clients.Client = clients.resolve_client()
config.apply()


__version__ = "0.5.0"
