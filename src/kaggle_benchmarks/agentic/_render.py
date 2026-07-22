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

"""Notebook-rendering helper for the experimental agentic package."""

from __future__ import annotations

from typing import Any


class PanelRenderable:
    """Mixin: makes a class with a ``__panel__`` method render in notebooks.

    Jupyter/IPython display a bare object via ``_repr_mimebundle_``; Panel's
    ``__panel__`` protocol alone isn't picked up for bare cell output (which is
    why ``kaggle_benchmarks.messages.Message`` grew a ``_repr_mimebundle_`` via
    ``rendering.RenderMixin``). This delegates to the Panel viewable that
    ``__panel__`` returns, so subclasses only need to define ``__panel__``.
    """

    def __panel__(self) -> Any:  # implemented by subclasses
        raise NotImplementedError

    def _repr_mimebundle_(self, include: Any = None, exclude: Any = None) -> Any:
        return self.__panel__()._repr_mimebundle_(include, exclude)
