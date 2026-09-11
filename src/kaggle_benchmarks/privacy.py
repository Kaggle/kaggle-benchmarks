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

"""What may be shown of a run from a private split.

A private run shows its id, status and timing. Its params, transcript,
result and any traceback are hidden.

Output is what needs guarding because it outlives the process: a notebook
saves its output into the .ipynb file and shares it, and terminal output
ends up in CI logs. A permission check added later cannot remove either.
That is why this applies everywhere, not only in notebooks.

Owners can see their own data by asking for it:

    kbench.reveal_private()          # for the rest of the session

    with kbench.reveal_private():    # or until the block ends
        display(runs.private)
"""

import contextlib
import logging
from typing import TYPE_CHECKING, Any

from kaggle_benchmarks._config import HostEnvironment, config, detect_host_environment

if TYPE_CHECKING:
    from kaggle_benchmarks import runs

logger = logging.getLogger(__name__)

MASK = "🔒"

HIDDEN_LABEL = f"{MASK} private row (hidden)"

HIDDEN_BODY = (
    "This row comes from the private split, so its params, transcript, result "
    "and errors are hidden. Showing them would save them into this notebook. "
    "Call `kbench.reveal_private()` to see them."
)

FAILURE_BODY = (
    "A row from the private split failed. The error is hidden because its "
    "message usually quotes the row. The original is on the run in "
    "`error_message` and in the run file. Call `kbench.reveal_private()` and "
    "run again to see it here."
)

_REVEAL_IN_NOTEBOOK_WARNING = (
    "Private rows will now be rendered into this notebook. Anything displayed "
    "from here on is saved into the .ipynb file and is shared along with it. "
    "Clear the cell outputs before sharing this notebook, or look at private "
    "rows from a terminal instead."
)


class PrivateRunError(Exception):
    """Replaces the exception a private row raised, so it is not printed.

    The original is still on the run and in the run file.
    """


# Both live in modules that import this one, so a top-level import would be a
# cycle. Resolved once and kept: these run in rendering and streaming loops,
# where an import per call was measurable (650ns a call, against 80ns).
_private_split: Any = None
_contexts: Any = None


def is_hidden(run: "runs.Run[Any]") -> bool:
    """Whether this run's contents must be kept out of rendered output."""
    global _private_split
    if _private_split is None:
        from kaggle_benchmarks.runs import Split

        _private_split = Split.PRIVATE
    return run.split is _private_split and not config.reveal_private


def context_is_hidden() -> bool:
    """Same as `is_hidden`, for the run currently executing.

    Live UI handlers render messages as they arrive, before any run object is
    returned.
    """
    global _contexts
    if _contexts is None:
        from kaggle_benchmarks import contexts

        _contexts = contexts
    run = _contexts.get_current().run
    return run is not None and is_hidden(run)


def count_hidden(runs_: "runs.Runs[Any]") -> int:
    return sum(1 for run in runs_ if is_hidden(run))


def guard_repr(cls: type) -> None:
    """Stops a dataclass repr from printing a hidden run.

    Python falls back to repr whenever no renderer is available: no Panel, a
    Panel that cannot draw, a plain REPL, a failing test, a log line. The
    generated repr prints every field, which for a run includes its params
    and its whole transcript.
    """
    generated: Any = cls.__repr__

    def safe(self: Any) -> str:
        if not is_hidden(self):
            return generated(self)
        return (
            f"{type(self).__name__}(task={self.task.name!r}, id={self.id!r}, "
            f"status={self.status}, split={self.split}, hidden)"
        )

    cls.__repr__ = safe  # type: ignore[assignment]


def banner(hidden: int) -> str:
    """The notice shown above a listing that is hiding some of its rows."""
    rows = "row is" if hidden == 1 else "rows are"
    return (
        f"{MASK} **{hidden} private {rows} hidden.** Their results still count "
        "toward the score. Call `kbench.reveal_private()` to see them, "
        "remembering that this writes them into the notebook's saved output."
    )


class _Revealed(contextlib.AbstractContextManager):
    """Restores the previous setting when used as a context manager."""

    def __init__(self, previous: bool):
        self._previous = previous

    def __exit__(self, *exc_details) -> None:
        config.reveal_private = self._previous


def reveal_private() -> _Revealed:
    """Shows private-split rows in rendered output.

    This takes effect immediately and stays on for the rest of the session.
    Used as a context manager, everything goes back to hidden when the block
    ends:

        kbench.reveal_private()

        with kbench.reveal_private():
            display(runs.private)

    The setting is process-wide, so with `n_jobs > 1` an open block reveals
    private rows for every thread, not only the caller's. It is meant for
    inspecting your own data by hand.

    In a notebook it logs a warning, since what it reveals is saved into the
    notebook file.
    """
    previous = config.reveal_private
    config.reveal_private = True
    if detect_host_environment() is not HostEnvironment.TERMINAL:
        logger.warning(_REVEAL_IN_NOTEBOOK_WARNING)
    return _Revealed(previous)


def hide_private() -> None:
    """Hides private-split rows again, undoing `reveal_private()`."""
    config.reveal_private = False
