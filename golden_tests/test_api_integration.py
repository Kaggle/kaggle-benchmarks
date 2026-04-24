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

import os
from typing import Generic, TypeVar

import openai
import pydantic
import pytest
from google import genai

from kaggle_benchmarks import chats, llm_messages, prompting, providers, utils
from kaggle_benchmarks import tools as tool_utils
from kaggle_benchmarks.actors import llms
from kaggle_benchmarks.content_types import images

http_client = utils.build_httpx_client("test_cache")


def create_openai_client(cls=providers.openai.OpenAIResponsesAPI, **kwargs):
    if "OPENAI_API_KEY" not in os.environ:
        pytest.skip("Missing OPENAI_API_KEY environment variable.")
    return cls(
        client=openai.OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            http_client=http_client,
        ),
        **kwargs,
    )


def create_google_client(cls=providers.genai.GoogleGenAI, **kwargs):
    if "GEMINI_API_KEY" not in os.environ:
        pytest.skip("Missing GEMINI_API_KEY environment variable.")
    return cls(
        client=genai.Client(api_key=os.environ["GEMINI_API_KEY"]),
        **kwargs,
    )


def create_model_proxy_openai_client(**kwargs):
    if "MODEL_PROXY_API_KEY" not in os.environ:
        pytest.skip("Missing MODEL_PROXY_API_KEY environment variable.")
    return providers.openai.ModelProxyOpenAI(
        client=openai.OpenAI(
            api_key=os.environ["MODEL_PROXY_API_KEY"],
            base_url=os.environ["MODEL_PROXY_URL"],
            http_client=http_client,
        ),
        **kwargs,
    )


def create_model_proxy_genai_client(**kwargs):
    if "MODEL_PROXY_API_KEY" not in os.environ:
        pytest.skip("Missing MODEL_PROXY_API_KEY environment variable.")
    return providers.genai.ModelProxyGenAI(
        client=genai.Client(
            api_key=os.environ["MODEL_PROXY_API_KEY"],
            http_options={
                "api_version": "v1",
                "base_url": os.environ["MODEL_PROXY_URL"].replace("/openapi", "/genai"),
            },
        ),
        **kwargs,
    )


PARAMS = [
    pytest.param(
        (
            create_openai_client,
            dict(
                model="gpt-4o",
                support_structured_outputs=True,
                support_tool_calling=True,
            ),
        ),
        id="openai[+s+t]",
    ),
    pytest.param(
        (
            create_openai_client,
            dict(
                model="gpt-4o",
                support_structured_outputs=True,
                support_tool_calling=False,
            ),
        ),
        id="openai[+s-t]",
    ),
    pytest.param(
        (
            create_openai_client,
            dict(
                model="gpt-4o",
                support_structured_outputs=False,
                support_tool_calling=False,
            ),
        ),
        id="openai[-s-t]",
    ),
    pytest.param(
        (
            create_openai_client,
            dict(model="o4-mini", cls=providers.openai.StreamingOpenAIResponsesAPI),
        ),
        id="openai-o4-mini-streaming",
    ),
    pytest.param(
        (create_google_client, dict(model="gemini-2.5-flash")),
        id="google-gemini-2.5-flash",
    ),
    pytest.param(
        (
            create_google_client,
            dict(model="gemini-2.5-flash", cls=providers.genai.StreamingGoogleGenAI),
        ),
        id="google-gemini-2.5-flash-streaming",
    ),
]

PROXY_MODELS = [
    "google/gemini-2.0-flash",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-pro",
    "google/gemini-3-flash-preview",
    "google/gemma-3-12b",
    "qwen/qwen3-235b-a22b-instruct-2507",
    "qwen/qwen3-next-80b-a3b-instruct",
    "anthropic/claude-haiku-4-5@20251001",
    "anthropic/claude-opus-4-5@20251101",
    "anthropic/claude-sonnet-4-5@20250929",
    "deepseek-ai/deepseek-r1-0528",
    "deepseek-ai/deepseek-v3.2",
    "zai/glm-5",
    # "google/gemini-3.1-flash-lite-preview",
]
for name in PROXY_MODELS:
    PARAMS.append(
        pytest.param(
            (create_model_proxy_openai_client, dict(model=name)),
            id=f"model-proxy-openai-{name}",
        )
    )

for name in PROXY_MODELS:
    PARAMS.append(
        pytest.param(
            (create_model_proxy_genai_client, dict(model=name)),
            id=f"model-proxy-genai-{name}",
        )
    )


@pytest.fixture
def llm(request):
    model_factory, params = request.param
    return model_factory(**params)


@pytest.mark.parametrize("llm", PARAMS, indirect=True)
def test_text_generation(llm):
    """Tests basic text generation to ensure the model responds."""
    with chats.new():
        response = llm.prompt("Say 'hello world' and nothing else.")
        assert "hello world" in response.lower()


class UserInfo(pydantic.BaseModel):
    name: str
    age: int


@pytest.mark.parametrize("llm", PARAMS, indirect=True)
def test_structured_output(llm):
    """Tests the model's ability to generate a simple Pydantic model."""
    with chats.new():
        response = llm.prompt(
            "Generate a user named Alice who is 30 years old.", schema=UserInfo
        )
        assert isinstance(response, UserInfo)
        assert response.name == "Alice"
        assert response.age == 30


class UserDetails(pydantic.BaseModel):
    user: UserInfo
    address: str


@pytest.mark.parametrize("llm", PARAMS, indirect=True)
def test_nested_structured_output(llm):
    """Tests the model's ability to generate a nested Pydantic model."""
    with chats.new():
        try:
            response = llm.prompt(
                "Generate a user named Alice who is 30 years old and lives at 123 Kaggle Street.",
                schema=UserDetails,
            )
            assert isinstance(response, UserDetails)
            assert response.user.name == "Alice"
            assert response.user.age == 30
            assert "123 Kaggle Street" in response.address
        except prompting.ResponseParsingError as e:
            pytest.xfail(
                f"Model {llm.model} may not support nested structured output: {e}"
            )


T = TypeVar("T")


class User(prompting.RenderablePydanticModel, Generic[T]):
    # class User(pydantic.BaseModel, Generic[T]):
    user: T
    address: str


@pytest.mark.parametrize("llm", PARAMS, indirect=True)
def test_generic_structured_output(llm):
    """Tests the model's ability to generate a nested Pydantic model."""
    with chats.new():
        try:
            response = llm.prompt(
                "Generate a user named Alice who is 30 years old and lives at 123 Kaggle Street.",
                schema=User[UserInfo],
            )
            assert isinstance(response, User)
            assert response.user.name == "Alice"
            assert response.user.age == 30
            assert "123 Kaggle Street" in response.address
        except prompting.ResponseParsingError as e:
            pytest.xfail(
                f"Model {llm.model} may not support nested structured output: {e}"
            )


@pytest.mark.parametrize("llm", PARAMS, indirect=True)
def test_vision_input(llm):
    """Tests the model's ability to process image input."""
    image = images.from_url(
        "https://storage.googleapis.com/kaggle-organizations/5154/thumbnail.png"
    )

    with chats.new("Vision Test Chat"):
        if llm.support_vision:
            response = llm.prompt("What is in this image?", image=image)
            assert "goose" in response.lower() or "bird" in response.lower()
        else:
            with pytest.raises(ValueError, match="Vision not supported"):
                llm.prompt("What is in this image?", image=image)


class StockPrice(pydantic.BaseModel):
    symbol: str
    price: float

    model_config = pydantic.ConfigDict(
        title="StockPrice",
        extra="forbid",
    )


def get_stock_price(symbol: str) -> float:
    """Gets the current stock price for a given symbol."""
    if symbol == "KGL":
        return 120.5
    elif symbol == "BNCH":
        return 210.3
    else:
        return 0.0


@pytest.mark.parametrize(
    "schema",
    [
        pytest.param(str, id="output_str"),
        pytest.param(float, id="output_primitive"),
        pytest.param(StockPrice, id="output_pydantic"),
    ],
)
@pytest.mark.parametrize("llm", PARAMS, indirect=True)
def test_tool_calling(llm, schema):
    """Tests the full tool-calling loop with various output schemas."""
    with chats.new() as chat:
        try:
            response = llm.prompt(
                "What is the price of KGL?",
                schema=schema,
                tools=[get_stock_price],
                max_tool_calls=2,
            )
        except prompting.ResponseParsingError as e:
            # Not all models reliably produce structured output, so we fail gracefully.
            pytest.xfail(str(e))
        except llms.ToolInvocationLimitExhausted as e:
            # TODO: some model will not see tool invocation and will call the same tool over and over
            pytest.xfail(str(e))

    assert isinstance(response, schema)
    # models may not respond correctly but should respond in proper format
    # if schema is float:
    #     assert response == 120.5
    # elif schema is StockPrice:
    #     assert response.price == 120.5
    #     assert "KGL" in response.symbol.upper()

    assert len(chat.messages) == 2
    llm_message = chat.messages[1]
    assert isinstance(llm_message, llm_messages.LLMMessage)
    assert llm_message.tool_calls is not None
    # This should be exactly one, but some models may generate more.
    assert len(llm_message.tool_calls) >= 1

    # The sub-chat containing the tool invocation should be preserved.
    assert llm_message.chat
    assert llm_message.chat.messages

    tool_call = llm_message.tool_calls[0]
    assert isinstance(
        tool_call, (tool_utils.ToolInvocation, tool_utils.ToolInvocationResult)
    )
    assert tool_call.name == "get_stock_price"
    assert "symbol" in tool_call.arguments
    assert "KGL" in tool_call.arguments["symbol"].upper()


@pytest.mark.parametrize("llm", PARAMS, indirect=True)
def test_parallel_tool_calling(llm):
    """Tests the LLM's ability to make parallel tool calls."""
    prompt = "What are the stock prices for 'KGL' and 'BNCH'?"
    with chats.new() as chat:
        response = llm.prompt(prompt, tools=[get_stock_price])
        llm_message = chat.messages[-1]

    assert isinstance(llm_message, llm_messages.LLMMessage)
    assert isinstance(response, str)
    assert llm_message.tool_calls is not None
    assert len(llm_message.tool_calls) >= 2

    tool_names = {call.name for call in llm_message.tool_calls}
    assert tool_names == {"get_stock_price"}

    symbols = {call.arguments["symbol"] for call in llm_message.tool_calls}
    assert symbols == {"KGL", "BNCH"}


@pytest.mark.parametrize("llm", PARAMS, indirect=True)
def test_tool_calling_structured_args(llm):
    """Tests tool calling where the tool argument is a Pydantic model."""

    class Point(pydantic.BaseModel):
        x: int
        y: int

    def draw_point(point: Point) -> str:
        """Draws a point on a canvas."""
        if isinstance(point, dict):
            point = Point.model_validate(point)
        return f"Drawing point at ({point.x}, {point.y})"

    with chats.new() as chat:
        try:
            response = llm.prompt(
                "Draw a point at (10, 20)",
                tools=[draw_point],
            )
        except prompting.ResponseParsingError as e:
            pytest.fail(str(e))
        except llms.ToolInvocationLimitExhausted as e:
            pytest.fail(str(e))

    assert isinstance(response, str)

    llm_message = chat.messages[1]
    assert llm_message.tool_calls
    tool_call = llm_message.tool_calls[0]
    assert tool_call.name == "draw_point"
    assert "point" in tool_call.arguments
    point = tool_call.arguments["point"]
    if isinstance(point, dict):
        assert point == {"x": 10, "y": 20}
    else:
        assert point.x == 10
        assert point.y == 20


def get_user_id(username: str) -> int:
    """Gets the user ID for a given username."""
    if username == "test_user":
        return 123
    else:
        return -1


def get_user_posts(user_id: int) -> list[str]:
    """Gets the posts for a given user ID."""
    if user_id == 123:
        return ["Post 1", "Post 2"]
    else:
        return []


@pytest.mark.parametrize("llm", PARAMS, indirect=True)
def test_dependent_tool_calling(llm):
    """Tests the LLM's ability to make dependent tool calls."""

    prompt = "What are the posts for user 'test_user'?"
    with chats.new() as chat:
        response = llm.prompt(prompt, tools=[get_user_id, get_user_posts])

    assert "Post 1" in response
    assert "Post 2" in response
    assert len(chat.messages) == 2
    llm_message = chat.messages[-1]
    assert isinstance(llm_message, llm_messages.LLMMessage)
    assert isinstance(response, str)
    assert llm_message.tool_calls is not None
    assert len(llm_message.tool_calls) >= 2

    tool_names = {call.name for call in llm_message.tool_calls}
    assert tool_names == {"get_user_id", "get_user_posts"}
