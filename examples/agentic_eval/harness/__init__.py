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

"""DEPRECATED compatibility shim.

The real implementation now lives in ``kaggle_benchmarks.agentic`` (built on the
library's own types). This ``harness`` package is kept only as a thin alias so
older imports keep working; ``harness`` also collides with a third-party package
name, so prefer importing from ``kaggle_benchmarks.agentic`` directly.
"""

from kaggle_benchmarks.agentic import *  # noqa: F401,F403
from kaggle_benchmarks.agentic.demo import *  # noqa: F401,F403
