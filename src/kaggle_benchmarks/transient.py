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

"""Telling "the provider was busy" apart from "the model got it wrong".

Nothing here fires on its own. A task opts in::

    @kbench.task(name="Counts characters")
    def character_count(llm):
        with transient.unmeasured():
            answer = llm.prompt("how many r in strawberry?")
        assertions.assert_equal(3, int(answer))

If the provider returns 503, the run ends ``UNMEASURED`` and the tasks around
it still run. Anything else raises as usual, so a bug in the task stays a bug.

The classifier reads status codes out of the exception rather than importing
each provider's error classes: kbench talks to several SDKs and a task may hold
a client from any of them.
"""

from __future__ import annotations

import contextlib
import re

from kaggle_benchmarks.tasks import Unmeasured

# 408 request timeout, 409 conflict, 425 too early, 429 rate limited,
# 500/502/503/504 upstream trouble. All mean "ask again later" rather than
# "your request was wrong".
_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

_STATUS_ATTRS = ("status_code", "code", "status", "http_status")

_MESSAGE_SIGNALS = (
    "unavailable",
    "overloaded",
    "resource_exhausted",
    "rate limit",
    "ratelimit",
    "too many requests",
    "deadline exceeded",
    "timed out",
    "timeout",
    "connection reset",
    "service unavailable",
    "temporarily unavailable",
)


def _status_of(error: BaseException) -> int | None:
    """Reads an HTTP status off an exception, whichever SDK raised it."""
    for attribute in _STATUS_ATTRS:
        value = getattr(error, attribute, None)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    # Fall back to the text: several SDKs only put the code in the message,
    # e.g. "ServerError: 503 UNAVAILABLE. {...}".
    match = re.search(r"\b([45]\d{2})\b", str(error))
    return int(match.group(1)) if match else None


def is_transient(error: BaseException) -> bool:
    """Whether ``error`` looks like the provider asking to be tried later.

    Deliberately conservative about what it reports as transient: calling a
    real failure transient quietly removes it from the score, which is worse
    than leaving a provider blip in.
    """
    if isinstance(error, TimeoutError):
        return True
    status = _status_of(error)
    if status is not None:
        return status in _RETRYABLE_STATUS
    text = str(error).lower()
    return any(signal in text for signal in _MESSAGE_SIGNALS)


@contextlib.contextmanager
def unmeasured(reason: str | None = None):
    """Turns a transient provider error inside the block into ``Unmeasured``.

    Args:
        reason: Recorded on the run in place of the exception text.

    Raises:
        Unmeasured: If the block raises and :func:`is_transient` recognises it.
        Exception: Anything else, unchanged.
    """
    try:
        yield
    except Exception as error:
        if not is_transient(error):
            raise
        raise Unmeasured(reason or f"{type(error).__name__}: {error}") from error
