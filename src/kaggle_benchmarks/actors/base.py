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

"""Concrete actors.

The abstract ``Actor`` base class lives in :mod:`kaggle_benchmarks.core`
(alongside the other core objects) and is re-exported here. This module
defines the concrete actor types and singletons.
"""

from kaggle_benchmarks.core import Actor

__all__ = ["Actor", "Tool", "assertion", "system", "user"]


class Tool(Actor):
    def __init__(self, name: str = "tool"):
        super().__init__(name=name, role="tool")


system = Actor(name="System", role="system", avatar="⚙️")
assertion = Actor(name="Assertion", role="system", avatar="🚨️")
user = Actor(name="User", role="user", avatar="👤")
