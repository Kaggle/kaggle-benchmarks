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

from kaggle_benchmarks.kaggle import serialization
from kaggle_benchmarks.kaggle.benchmark_types_pb2 import (  # type: ignore[attr-defined]
    BenchmarkTaskRun,
    BenchmarkTaskVersion,
)
from kaggle_benchmarks.kaggle.client import KaggleClient
from kaggle_benchmarks.kaggle.model_proxy import ModelProxy
from kaggle_benchmarks.kaggle.models import (
    is_configured,
    load_available_models,
    load_default_model,
    load_judge_model,
    load_model,
)
