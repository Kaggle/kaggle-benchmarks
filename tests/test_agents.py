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

import pytest

from kaggle_benchmarks import chats
from kaggle_benchmarks.agents import LLMActor, LLMAgent
from kaggle_benchmarks.core import Session
from kaggle_benchmarks.llm_messages import LLMMessage
from kaggle_benchmarks.tools.base import ToolInvocation


class ScriptedAgent(LLMAgent):
    """An agent whose provider call replays canned responses, one per invoke."""

    def __init__(self, responses, cycle=False, **kwargs):
        super().__init__(**kwargs)
        self._responses = itertools.cycle(responses) if cycle else iter(responses)
        self.invocations = []

    def invoke(self, messages, *, system=None, schema=str, tools=None, **kwargs):
        self.invocations.append((messages, tools))
        response = next(self._responses)
        return LLMMessage(
            sender=self,
            content=response.get("content", ""),
            tool_calls=response.get("tool_calls"),
        )


def _tool_call(name, arguments, call_id="c1"):
    return [ToolInvocation(name=name, arguments=arguments, call_id=call_id)]


def test_run_answers_in_its_own_session():
    agent = ScriptedAgent([{"content": "42."}], name="Answerer")

    with chats.new("outer") as outer:
        answer = agent.run("What is 6 times 7?")

    assert "42" in answer
    # The agent's turns live in its own session, nested in the caller's history.
    nested = [e for e in outer.history if isinstance(e, Session)]
    assert [s.name for s in nested] == ["Answerer"]


def test_instructions_are_sent_once_per_session():
    agent = ScriptedAgent(
        [{"content": "a"}, {"content": "b"}],
        name="Guided",
        instructions="Be terse.",
    )

    with chats.new("outer"):
        agent.run("first")
        agent.run("second")

    texts = [m.text for m in agent._session.messages]
    assert texts.count("Be terse.") == 1


def test_a_second_run_keeps_the_earlier_exchange():
    """A subagent has a conversation, not just a return value."""
    agent = ScriptedAgent([{"content": "Alice."}, {"content": "Alice."}], name="Memo")

    with chats.new("outer"):
        agent.run("My name is Alice. Who am I?")
        first_session = agent._session
        agent.run("Who am I?")

    assert agent._session is first_session
    assert "My name is Alice. Who am I?" in [m.text for m in agent._session.messages]


def test_reset_starts_a_fresh_session():
    agent = ScriptedAgent([{"content": "a"}, {"content": "b"}], name="Memo")

    with chats.new("outer"):
        agent.run("first")
        first = agent._session
        agent.reset()
        agent.run("second")

    assert agent._session is not first
    assert "first" not in [m.text for m in agent._session.messages]


def test_tools_and_subagents_are_both_callable():
    def lookup(city: str) -> int:
        """Looks up a population."""
        return 100

    child = ScriptedAgent([{"content": "child answer"}], name="Child")
    parent = ScriptedAgent(
        [{"content": "done"}], name="Parent", tools=[lookup], subagents=[child]
    )

    names = [c.__name__ for c in parent.callables]
    assert names == ["lookup", "ask_child"]


def test_subagent_is_described_to_the_parent_for_tool_choice():
    child = ScriptedAgent(
        [{"content": "x"}], name="Chess Expert", description="Answers chess questions."
    )
    tool = child.as_tool()

    assert tool.__name__ == "ask_chess_expert"
    assert tool.__doc__ == "Answers chess questions."


def test_delegating_runs_the_subagent_and_nests_its_session():
    child = ScriptedAgent([{"content": "Paris."}], name="Geo")
    parent = ScriptedAgent(
        [
            {
                "content": "",
                "tool_calls": _tool_call("ask_geo", {"request": "capital?"}),
            },
            {"content": "The capital is Paris."},
        ],
        name="Parent",
        subagents=[child],
    )

    with chats.new("outer"):
        answer = parent.run("What is the capital of France?")

    assert "Paris" in answer
    # The child's turns are a branch of the parent's session, not flattened text.
    nested = [e for e in parent._session.history if isinstance(e, Session)]
    assert [s.name for s in nested] == ["Geo"]
    assert "capital?" in [m.text for m in child._session.messages]


def test_an_agent_is_an_actor_not_an_llmchat():
    """The new surface stands on its own; LLMChat is untouched by it."""
    from kaggle_benchmarks.actors import LLMChat
    from kaggle_benchmarks.core import Actor

    agent = ScriptedAgent([{"content": "x"}], name="Standalone")
    assert isinstance(agent, LLMActor)
    assert isinstance(agent, Actor)
    assert not isinstance(agent, LLMChat)


def test_an_actor_without_tools_is_just_a_model_call():
    """LLMAgent with nothing attached behaves as a plain LLMActor would."""
    agent = ScriptedAgent([{"content": "Paris."}], name="Bare")

    with chats.new("outer"):
        assert "Paris" in str(agent.run("Capital of France?"))
    assert agent.callables == []


def test_genai_agent_talks_to_gemini_directly(monkeypatch):
    """The agent builds a direct client, not one pointed at Model Proxy."""
    from kaggle_benchmarks.agents import GenAIAgent

    monkeypatch.setenv("MODEL_PROXY_URL", "https://proxy.example/genai")
    monkeypatch.setenv("MODEL_PROXY_API_KEY", "proxy-key")

    agent = GenAIAgent("gemini-2.5-flash", api_key="direct-key")

    assert agent.model == "gemini-2.5-flash"
    assert agent.name == "gemini-2.5-flash"
    assert "proxy.example" not in str(agent.client._api_client._http_options.base_url)


def test_genai_agent_requires_a_key(monkeypatch):
    from kaggle_benchmarks.agents import GenAIAgent

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)

    with pytest.raises(ValueError, match="No Google API key"):
        GenAIAgent("gemini-2.5-flash")


def test_genai_agent_is_an_agent_first(monkeypatch):
    """Tools and subagents compose exactly as they do on the base class."""
    from kaggle_benchmarks.agents import GenAIAgent

    monkeypatch.setenv("GOOGLE_API_KEY", "env-key")

    def get_weather(city: str) -> str:
        """Looks up the weather."""
        return "sunny"

    child = GenAIAgent("gemini-2.5-flash", name="Local", description="Knows Zurich.")
    parent = GenAIAgent("gemini-2.5-flash", tools=[get_weather], subagents=[child])

    assert [c.__name__ for c in parent.callables] == ["get_weather", "ask_local"]


# --- ADKAgent -------------------------------------------------------------
# google-adk is an optional extra, so these cover everything reachable without
# it: configuration, the composition behaviour, the install hint, and the
# translation of ADK events into a session.


def _part(**kwargs):
    from types import SimpleNamespace

    kwargs.setdefault("function_call", None)
    kwargs.setdefault("function_response", None)
    kwargs.setdefault("text", None)
    kwargs.setdefault("thought", False)
    return SimpleNamespace(**kwargs)


def _event(*parts):
    from types import SimpleNamespace

    return SimpleNamespace(content=SimpleNamespace(parts=list(parts)))


def test_adk_agent_is_configurable_without_adk_installed():
    """Importing and constructing must not require the optional dependency."""
    from kaggle_benchmarks.agents import ADKAgent

    def get_weather(city: str) -> str:
        """Looks up the weather."""
        return "sunny"

    child = ADKAgent(name="Local", description="Knows Zurich.")
    agent = ADKAgent("gemini-2.5-flash", tools=[get_weather], subagents=[child])

    assert agent.model == "gemini-2.5-flash"
    assert [c.__name__ for c in agent.callables] == ["get_weather", "ask_local"]


def test_adk_agent_reports_the_missing_dependency_clearly():
    from kaggle_benchmarks.agents import ADKAgent

    pytest.importorskip  # noqa: B018 - documents intent; we want the failure path
    try:
        import google.adk  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("google-adk is installed, so the hint path can't be reached")

    with pytest.raises(ImportError, match="pip install google-adk"):
        ADKAgent().run("hi")


def test_adk_agent_rejects_structured_output_for_now():
    from kaggle_benchmarks.agents import ADKAgent

    with pytest.raises(NotImplementedError):
        ADKAgent().run("hi", schema=int)


def test_recording_adk_events_writes_text_into_the_session():
    from kaggle_benchmarks.agents.adk import record_event

    agent = ADKAgentStub()
    with chats.new("outer"):
        with agent.session():
            answer = record_event(_event(_part(text="The capital is Paris.")), agent)
            texts = [m.text for m in agent._session.messages]

    assert answer == "The capital is Paris."
    assert "The capital is Paris." in texts


def test_recording_adk_events_maps_tool_calls_and_results():
    from types import SimpleNamespace

    from kaggle_benchmarks.agents.adk import record_event

    agent = ADKAgentStub()
    call = SimpleNamespace(name="get_weather", args={"city": "Zurich"})
    result = SimpleNamespace(name="get_weather", response="sunny")

    with chats.new("outer"):
        with agent.session():
            record_event(_event(_part(function_call=call)), agent)
            record_event(_event(_part(function_response=result)), agent)
            messages = agent._session.messages

    invoked = [c for m in messages for c in (m.tool_calls or [])]
    assert [c.name for c in invoked] == ["get_weather"]
    assert invoked[0].arguments == {"city": "Zurich"}
    assert any("sunny" in str(m.content) for m in messages)


def test_recording_adk_events_keeps_thoughts_out_of_the_answer():
    from kaggle_benchmarks.agents.adk import record_event

    agent = ADKAgentStub()
    with chats.new("outer"):
        with agent.session():
            answer = record_event(
                _event(_part(text="thinking...", thought=True)), agent
            )
            traces = [m.reasoning_traces for m in agent._session.messages]

    assert answer is None  # a thought is not the reply
    assert "thinking..." in traces


class ADKAgentStub(LLMAgent):
    """An ADK-shaped agent that never reaches ADK, for the mapping tests."""

    def __init__(self, **kwargs):
        kwargs.setdefault("name", "Stub")
        super().__init__(**kwargs)
