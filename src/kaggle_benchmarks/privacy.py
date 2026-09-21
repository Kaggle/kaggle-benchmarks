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

"""Hides the rows of a hidden split from output and from their parent run.

While such a row runs:
- no event reaches a listener, so UI handlers print nothing for it;
- a failure raises `HiddenRunError` instead of the original error;
- the run is not attached to its parent run, so displaying or serializing
  the parent leaves it out.

`contexts.hidden()` opens the scope; `Task.evaluate_splits` is the supported
way to use it.
"""

import contextlib
import logging
import warnings

from kaggle_benchmarks._config import HostEnvironment, config, detect_host_environment

logger = logging.getLogger(__name__)

FAILURE_MESSAGE = (
    "A row of a hidden split failed. The error is withheld because it usually "
    "contains the row. See `run.error_message` or the run file, or call "
    "`kbench.reveal_hidden()` and run again."
)

_REVEAL_WARNING = "Hidden rows will be displayed and saved in the notebook."

FIRST_USE_WARNING = (
    "Hidden splits are experimental. Private rows don't appear in output or in "
    "the main task's run file, but each still writes its own run file to the "
    "working directory. Run `%choose <main task>` before publishing to delete them."
)

_warned = False


def warn_first_use() -> None:
    """Warns once per session, at the caller of `evaluate_splits`."""
    global _warned
    if not _warned:
        _warned = True
        warnings.warn(FIRST_USE_WARNING, UserWarning, stacklevel=3)


class HiddenRunError(Exception):
    """Raised instead of the original error of a row in a hidden split."""


def output_hidden() -> bool:
    """Whether output is hidden for the current run."""
    from kaggle_benchmarks import contexts  # contexts imports this module.

    return contexts.get_current().hidden and not config.reveal_hidden


class _Revealed(contextlib.AbstractContextManager):
    def __init__(self, previous: bool):
        self._previous = previous

    def __exit__(self, *exc_details) -> None:
        config.reveal_hidden = self._previous


def reveal_hidden() -> _Revealed:
    """Shows the output of hidden rows while they run.

    Lasts until the end of a `with` block, or for the session; set
    `kbench.config.reveal_hidden = False` to undo. Applies to all threads.
    Hidden rows still stay out of their parent run.
    """
    previous = config.reveal_hidden
    config.reveal_hidden = True
    if detect_host_environment() is not HostEnvironment.TERMINAL:
        logger.warning(_REVEAL_WARNING)
    return _Revealed(previous)
