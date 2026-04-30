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


import panel as pn
import pytest

from kaggle_benchmarks import actors, chats, llm_messages, tools, ui


def test_render_thread():
    goose = actors.Actor("Gerald", "user", "🪿")
    cat = actors.Actor("Mittens", "user", "🐈")

    chat = chats.Chat(
        history=[
            chats.Message("Chirp chirp", sender=goose),
            chats.Message("Meow!", sender=cat),
        ]
    )
    rendered = ui.panel.render_chat_to_html(chat)

    assert "Chirp chirp" in rendered
    assert goose.name in rendered

    assert "Meow!" in rendered
    assert cat.name in rendered


LLM_ACTOR = actors.LLMChat()
USER_ACTOR = actors.Actor("user", "user", "🧑")
TOOL_INVOCATION = tools.ToolInvocation(name="test_tool", arguments={"arg1": "val1"})
TOOL_RESULT = tools.ToolInvocationResult(
    name="test_tool", arguments={"arg1": "val1"}, output="tool output"
)
CHAT_HISTORY = chats.Chat(
    history=[
        chats.Message("Hello", sender=USER_ACTOR),
        chats.Message("Hi", sender=LLM_ACTOR),
    ]
)
QUESTION_HISTORY = chats.Chat(
    history=[
        chats.Message("Question?", sender=USER_ACTOR),
    ]
)
USAGE = llm_messages.Usage(input_tokens=10, output_tokens=20)
COMPLEX_USAGE = llm_messages.Usage(input_tokens=100, output_tokens=50)


@pytest.mark.parametrize(
    "message",
    [
        pytest.param(
            llm_messages.LLMMessage(sender=LLM_ACTOR, content="Hello, world!"),
            id="simple_message",
        ),
        pytest.param(
            llm_messages.LLMMessage(
                sender=LLM_ACTOR,
                content="Thinking about something...",
                reasoning_traces="hmmm...",
            ),
            id="with_thinking",
        ),
        pytest.param(
            llm_messages.LLMMessage(
                sender=LLM_ACTOR, content="Some content", usage=USAGE
            ),
            id="with_usage",
        ),
        pytest.param(
            llm_messages.LLMMessage(
                sender=LLM_ACTOR,
                content="Using a tool",
                tool_calls=[TOOL_INVOCATION],
            ),
            id="with_tool_invocation",
        ),
        pytest.param(
            llm_messages.LLMMessage(
                sender=LLM_ACTOR, content="A tool was used", tool_calls=[TOOL_RESULT]
            ),
            id="with_tool_result",
        ),
        pytest.param(
            llm_messages.LLMMessage(
                sender=LLM_ACTOR, content="This is a summary.", chat=CHAT_HISTORY
            ),
            id="with_chat_history",
        ),
        pytest.param(
            llm_messages.LLMMessage(
                sender=LLM_ACTOR,
                content="Complex message",
                reasoning_traces="Thinking hard...",
                usage=COMPLEX_USAGE,
                tool_calls=[TOOL_INVOCATION],
                chat=QUESTION_HISTORY,
            ),
            id="all_parameters",
        ),
    ],
)
def test_render_llm_message_combinations(message):
    rendered = ui.panel.render_llm_message(message)
    assert isinstance(rendered, pn.chat.ChatMessage)
