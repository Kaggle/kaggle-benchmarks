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
# # Using a Judge LLM for Complex Assertions
#
# This example demonstrates how to use `assess_response_with_judge` to evaluate an LLM's response against a set of complex, qualitative criteria.

# %%

import dataclasses
import textwrap
from typing import Iterable, List

import kaggle_benchmarks as kbench

# %% [markdown]
# ---
# ### Example with Default `AssessReport`
#
# This first example uses the default `AssessReport` schema to evaluate an explanation of a technical concept.


# %%
@kbench.task()
def explain_zero_knowledge_proofs(llm):
    """Task to evaluate an explanation of Zero-Knowledge Proofs."""
    response = llm.prompt("Explain Zero-Knowledge Proofs (ZKPs).")

    assess_report = kbench.assertions.assess_response_with_judge(
        criteria=(
            # Core Definition
            "The explanation must state that the Prover validates a claim to the Verifier without revealing the underlying secret or data.",
            # Theoretical Pillars
            "The explanation must explicitly mention the three formal properties: Completeness, Soundness, and Zero-Knowledge.",
            # Nuance 1: Probabilistic Nature
            "The explanation must clarify that ZKPs often rely on probabilistic verification.",
            # Nuance 2: Types of ZKP
            "The explanation must distinguish between Interactive and Non-Interactive ZKPs.",
            # Pedagogical Element
            "The response should include a conceptual analogy (e.g., Ali Baba's Cave).",
        ),
        response_text=response,
        judge_llm=kbench.judge_llm,
    )

    if assess_report is None:
        kbench.assertions.assert_fail(
            expectation="Judge LLM should respond with an assessment report."
        )
    else:
        for result in assess_report.results:
            kbench.assertions.assert_true(
                result.passed,
                expectation=f"Should pass criterion: {result.criterion} with confidence: {result.confidence} and reason: {result.reason}",
            )


explain_zero_knowledge_proofs.run(kbench.llm)


# %% [markdown]
# ---
# ### Example with Custom `prompt_fn` and `output_schema`
#
# You can provide a custom prompt and output schema to tailor the judge's evaluation to your specific needs.


# %%
@dataclasses.dataclass
class StoryCritique:
    """A custom schema for receiving a story critique from the judge."""

    overall_rating: int  # A rating from 1 (poor) to 5 (excellent).
    feedback: str  # General feedback on the story.
    passed_checks: List[str]  # A list of criteria that the story successfully met.


def custom_story_prompt(criteria: Iterable[str], response_text: str) -> str:
    """A custom prompt function that instructs the judge to use the StoryCritique schema."""
    formatted_criteria = "\n".join(f"- {c}" for c in criteria)
    return textwrap.dedent(f"""
        You are a literary critic. Evaluate the following short story based on the provided criteria.

        **Short Story:**
        {response_text}

        **Evaluation Criteria:**
        {formatted_criteria}

        Provide your assessment in a JSON format with three fields:
        1. `overall_rating`: An integer from 1 to 5.
        2. `feedback`: A short paragraph of constructive feedback.
        3. `passed_checks`: A list of strings containing the criteria that were fully met.
    """)


@kbench.task()
def critique_short_story(llm):
    """This task demonstrates using a custom prompt and schema for the judge."""
    story = llm.prompt(
        "Write a one-paragraph short story about a robot who discovers music."
    )

    critique = kbench.assertions.assess_response_with_judge(
        criteria=[
            "The story is exactly one paragraph.",
            "The main character is a robot.",
            "The story is about the discovery of music.",
            "The story has a clear beginning, middle, and end.",
            "The story is inspired by a true story.",  # This should FAIL.
        ],
        response_text=story,
        judge_llm=kbench.judge_llm,
        prompt_fn=custom_story_prompt,
        output_schema=StoryCritique,
    )

    if critique is None:
        kbench.assertions.assert_fail(
            expectation="Judge LLM should respond with a critique."
        )
    else:
        # You can now add assertions based on your custom critique object.
        kbench.assertions.assert_true(
            critique.overall_rating >= 3,
            f"Story rating was {critique.overall_rating}, which is below the acceptable threshold of 3.",
        )
        kbench.assertions.assert_true(
            "The story is inspired by a true story." not in critique.passed_checks,
            "The critique incorrectly passed a failing criterion.",
        )


critique_short_story.run(kbench.llm)

# %%
