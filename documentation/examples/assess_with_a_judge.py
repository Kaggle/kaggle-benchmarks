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

# %%
import kaggle_benchmarks as kbench


# %%
@kbench.task()
def explain_zero_knowledge_proofs(llm):
    # The prompt asks for a definition of a complex cryptographic concept
    response = llm.prompt(
        "Explain the mechanism and purpose of Zero-Knowledge Proofs (ZKPs)."
    )

    kbench.assertions.assess_response_with_judge(
        criteria=(
            # Core Definition
            "The explanation must state that the Prover validates a claim to the Verifier without revealing the underlying secret or data.",
            # Theoretical Pillars
            "The explanation must explicitly mention the three formal properties: Completeness, Soundness, and Zero-Knowledge.",
            # Nuance 1: Probabilistic Nature
            "The explanation must clarify that ZKPs often rely on probabilistic verification (the chance of cheating becomes negligible after many rounds) rather than a single deterministic check.",
            # Nuance 2: Types of ZKP
            "The explanation must distinguish between Interactive ZKPs (requires back-and-forth communication) and Non-Interactive ZKPs (succinct, single-message proofs like zk-SNARKs).",
            # Pedagogical Element
            "The response should include a conceptual analogy to explain the logic (e.g., Ali Baba's Cave, the color-blind friend, or Where's Waldo).",
        ),
        response_text=response,
        judge_llm=kbench.judge_llm,
    )


# %%
explain_zero_knowledge_proofs.run(kbench.llm)

# %%
