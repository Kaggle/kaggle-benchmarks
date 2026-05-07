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

import itertools
import json

from kaggle_benchmarks import actors
from kaggle_benchmarks.llm_messages import LLMMessage


class MockedChat(actors.LLMChat):
    """A mock LLMChat that returns pre-configured responses.

    Note: invoke() returns LLMMessage (not LLMResponse), exercising a
    different branch in LLMChat.respond() than real backends.
    """

    def __init__(
        self, responses: list[LLMMessage[str]], name="MockedChat", cycle=False, **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.response = itertools.cycle(responses) if cycle else iter(responses)
        self.invocations = []

    @classmethod
    def from_contents(cls, contents: list[str], cycle=False, **kwargs):
        return cls(
            responses=[
                LLMMessage(sender=None, content=content) for content in contents
            ],
            cycle=cycle,
            **kwargs,
        )

    @classmethod
    def from_contents_data(cls, contents: list[dict], cycle=False, **kwargs):
        return cls(
            responses=[
                LLMMessage(sender=None, content=json.dumps(content))
                for content in contents
            ],
            cycle=cycle,
            **kwargs,
        )

    @staticmethod
    def make_tool_call_response(
        name: str,
        arguments: dict | None = None,
        call_id: str = "call_1",
    ) -> LLMMessage[str]:
        """Creates an LLMMessage that simulates a tool call from the LLM.

        Sets tool_calls in _meta to match the dict format that real
        backends produce (normalised to OpenAI-style dicts).
        """
        msg = LLMMessage(sender=None, content="")
        msg._meta["tool_calls"] = [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        ]
        return msg

    def invoke(self, messages, tools=None, **kwargs):
        self.invocations.append((messages, kwargs))
        try:
            response = next(self.response)
            response.sender = self
            return response
        except StopIteration:
            assert False, "No more responses available"
