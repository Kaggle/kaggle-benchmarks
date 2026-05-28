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

from typing import Iterable, Literal, TypeVar

from kaggle_benchmarks import utils
from kaggle_benchmarks.messages import Message

T = TypeVar("T")


class Actor:
    def __init__(
        self,
        name: str | None = None,
        role: Literal["system", "user", "assistant", "developer", "tool"] = "assistant",
        avatar: str = "",
        id: str | None = None,
    ):
        self.name = type(self).__name__ if name is None else name
        self.id = id or name
        self.role = role
        self.avatar = avatar

    def send(
        self, message: T | Message[T], is_visible_to_llm: bool = True
    ) -> Message[T]:
        return self._send(message, is_visible_to_llm)

    def _send(self, message: T | Message[T], is_visible_to_llm: bool) -> Message[T]:
        from kaggle_benchmarks import chats

        if isinstance(message, Message):
            message.is_visible_to_llm = is_visible_to_llm
        else:
            message = Message(
                sender=self, content=message, is_visible_to_llm=is_visible_to_llm
            )

        chats.send(message)
        return message

    def stream(self, content: Iterable[str]) -> Message[str]:
        msg = Message(content="", sender=self, _status=utils.Status.RUNNING)
        self.send(msg)
        msg.stream(content)
        msg.status = utils.Status.SUCCESS
        return msg

    def as_dict(self) -> dict[str, str]:
        """Returns a dictionary representation of the agent."""
        return dict(id=self.id, name=self.name, role=self.role, avatar=self.avatar)

    def __repr__(self) -> str:
        cls = type(self)
        return f"{cls.__name__}(name={self.name!r}, avatar={self.avatar!r})"

    def __str__(self) -> str:
        return f"{self.avatar} {self.name}"

    def say(self, message: str) -> str:
        """Post a scripted message to the active ChatRoom.

        This is a **ChatRoom-only** method for code-driven actors (narrators,
        game engines, moderators) to inject predetermined content into the
        room's ground-truth log. The message is posted as-is — no LLM
        generation occurs.

        For LLM-generated responses, see ``LLMChat.reply()``.

        Args:
            message: The exact text to post to the room.

        Returns:
            The message content that was posted.

        Raises:
            RuntimeError: If called outside of an active ChatRoom context.
        """
        from kaggle_benchmarks import chats

        chat = chats.get_current_chat()
        if not isinstance(chat, chats.ChatRoom):
            raise RuntimeError(
                "Actor.say() must be called within an active ChatRoom context."
            )

        msg = Message(sender=self, content=message)
        chat.append(msg)
        return message


class Tool(Actor):
    def __init__(self, name: str = "tool"):
        super().__init__(name=name, role="tool")


system = Actor(name="System", role="system", avatar="⚙️")
assertion = Actor(name="Assertion", role="system", avatar="🚨️")
user = Actor(name="User", role="user", avatar="👤")
