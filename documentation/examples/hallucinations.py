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

# %% [markdown]
# ---
# title: Halluciantions
# ---
# %%

import pandas as pd
from pydantic import BaseModel

from kaggle_benchmarks import LLMChat, chats, judge_llm, llm, llms, task

# %%


@task()
def fake_story(llm: LLMChat, critic: LLMChat, question: str, topic: str) -> bool:
    answer = llm.prompt(question)

    with chats.new("Checking for hallucination"):
        result = critic.prompt(
            f"""I've asked a friend about a {topic}: {question}
It is clearly fake and has never happened, but did my friend spot that? Answer: {answer}""",
            schema=bool,
        )
        return result


# %%

df = pd.DataFrame(
    {
        "question": [
            "Describe in 100 words the epic rap battle between William Shakespeare and Marie Curie.",
            "Tell me about the time Albert Einstein and Napoleon Bonaparte worked on blockchain technology together.",
        ]
    }
).assign(topic="fictional historical event")
trials = fake_story.evaluate(llm=[llm], critic=[judge_llm], evaluation_data=df)
trials

# %%


class Story(BaseModel):
    question: str
    topic: str


class Questions(BaseModel):
    questions: list[Story]


@task()
def hallucination_competition(llm, llms):
    response = llm.prompt(
        "Write 10 question that you think will make LLM hallucinate.", schema=Questions
    )
    df = pd.DataFrame([q.model_dump() for q in response.questions])

    return (
        1
        - fake_story.evaluate(llm=llms, critic=[llm], evaluation_data=df)
        .as_dataframe()
        .result.mean()
    )


hallucination_competition.run(llm=llm, llms=llms.values())

# %%
