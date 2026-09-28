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

import functools


class EventManager:
    def __init__(self):
        self.listeners = []

    def clear(self):
        self.listeners = []

    def bind(self, listener):
        if listener not in self.listeners:
            self.listeners.append(listener)

    def unbind(self, listener):
        if listener in self.listeners:
            self.listeners.remove(listener)

    def dispatch(self, event, *args, **kwargs):
        for listener in list(self.listeners):
            if hasattr(listener, event):
                getattr(listener, event)(*args, **kwargs)

    def event_dispatcher(self, *events):
        """Wraps a method so that, after it runs, each named event is dispatched.

        Multiple event names may be given; they are dispatched in order with the
        same arguments. This is used to emit a legacy alias alongside the
        canonical event name (e.g. ``"new_event"`` and ``"new_message"``).
        """

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                for event in events:
                    self.dispatch(event, *args, **kwargs)
                return result

            return wrapper

        return decorator


manager = EventManager()
