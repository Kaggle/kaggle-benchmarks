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

"""Backward-compatible alias for the public ``ScriptedLLM``.

``MockedChat`` now lives in :mod:`kaggle_benchmarks.testing` as ``ScriptedLLM``.
This thin subclass preserves the original signature (positional ``name``) for
existing tests; new code should use ``kaggle_benchmarks.testing.ScriptedLLM``.
"""

from kaggle_benchmarks.testing import ScriptedLLM


class MockedChat(ScriptedLLM):
    """Deprecated alias for :class:`kaggle_benchmarks.testing.ScriptedLLM`."""

    def __init__(self, responses, name="MockedChat", cycle=False, **kwargs):
        super().__init__(responses, name=name, cycle=cycle, **kwargs)
