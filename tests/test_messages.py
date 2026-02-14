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

import json
from dataclasses import dataclass

import pydantic
import pytest

from kaggle_benchmarks import chats, messages, user
from tests.mocks import MockedChat


def test_raw_payload():
    float_response = '{"value": 0.01}'
    p = MockedChat.from_contents([float_response, '{"value": true}'])
    with chats.new() as chat:
        m = p.prompt(float_response, schema=float)
        assert m == 0.01
        assert "0.01" in chat.messages[-1].payload

    r = p.prompt('{"value": true}', schema=bool)
    assert r


def test_dataclass_payload():
    @dataclass
    class Point:
        x: float
        y: float

    msg = messages.Message(Point(1, 2), sender=user)
    assert json.loads(msg.payload) == {"x": 1, "y": 2}


def test_pydantic_payload():
    class Point(pydantic.BaseModel):
        x: float
        y: float

    msg = messages.Message(Point(x=1.5, y=2.5), sender=user)
    assert json.loads(msg.payload) == {"x": 1.5, "y": 2.5}


def test_class_payload():
    class Point:
        def __init__(self, x):
            self.x = x

    msg = messages.Message(Point(1), sender=user)
    with pytest.warns():
        payload = msg.payload

    assert json.loads(payload) == {"x": 1}
