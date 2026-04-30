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

from kaggle_benchmarks import chats, messages, utils


class ConsoleUI:
    def __init__(self, tab_size: int = 2):
        self.depth = 0
        self.tab_size = tab_size

    def new_chat(self, chat: chats.Chat):
        print(chat.__str__(indent=" " * (self.depth * self.tab_size)), end="")
        self.depth += 1

    def end_chat(self, chat: chats.Chat):
        self.depth -= 1

    def new_message(self, chat: chats.Chat, message: messages.Message):
        if isinstance(message, chats.Chat):
            return
        print(
            message.__str__(indent=" " * (self.depth * self.tab_size)),
            end="" if message.status == utils.Status.RUNNING else "\n",
        )

    def new_chunk(self, message: messages.Message, token: str):
        print(token, end="")

    def end_content(self, message: messages.Message):
        print()
