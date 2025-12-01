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

"""Provides context management for stateful interactions.

This module provides context management for internal framework use, not for
end-users defining benchmarks. It tracks the current chat, task, run,
and client.

The context can be viewed as the current state of a state machine, facilitating
management of complex interactions and state transitions.

A "global" context simplifies usage by avoiding explicit context passing to
functions and client code. This enables convenient shortcuts like `send(message)`,
which operates within the current conversation.

Example:

with context.enter(chat=chats.Chat()) as ctx:
    # Messages sent with 'chats.send' are scoped to this context.
    ...

"""

import contextlib
import dataclasses
from contextvars import ContextVar
from typing import Iterator, Self

from kaggle_benchmarks import chats, events, runs, utils
from kaggle_benchmarks.tasks import NonRecoverableError


@dataclasses.dataclass(frozen=True)
class Context:
    chat: chats.Chat = dataclasses.field(default_factory=chats.Chat)
    run: runs.Run | None = None
    parent: Self | None = None


_global = Context(chat=chats.GoldfishChat())
_current: ContextVar[Context] = ContextVar("current", default=_global)


@contextlib.contextmanager
def enter(**kwargs) -> Iterator[Context]:
    """Creates a new context based on the current context, overriding the
    specified keyword arguments.
    """
    # Thread-safe: Each thread/coroutine works with a copy of the context.
    from kaggle_benchmarks._config import config

    parent = _current.get()
    new = dataclasses.replace(parent, **kwargs, parent=parent)
    token = _current.set(new)
    try:
        for key, value in kwargs.items():
            events.manager.dispatch(f"new_{key}", value)
            value.status = utils.Status.RUNNING

        status = utils.Status.SUCCESS
        yield new

    except (NonRecoverableError, KeyboardInterrupt):
        raise
    except Exception:
        status = utils.Status.FAILED
        # Re-raise it if we are within a run so the exception can be propagated.
        if parent and parent.run:
            raise
        # At root run, optionally re-raise based on config.
        elif not config.continue_with_exceptions:
            raise

    finally:
        for key, value in kwargs.items():
            value.status = status
            events.manager.dispatch(f"end_{key}", value)

        _current.reset(token)


def set(**kwargs):
    """Creates a new context based on the current context, overriding the
    specified keyword arguments. This does not create a new scope like `enter`,
    but rather changes the current context in-place.
    """
    parent = _current.get()
    new = dataclasses.replace(parent, **kwargs, parent=parent)
    _current.set(new)


def get_current() -> Context:
    """Retrieves the current context."""
    return _current.get()


def set_from_context(context: Context):
    """
    Sets the current thread's context from a provided Context object.
    This is used to propagate context to worker threads.
    """
    _current.set(context)
