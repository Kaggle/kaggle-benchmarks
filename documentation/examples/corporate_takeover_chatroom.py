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

"""Corporate Takeover: Hostile Takeover with a "White Knight"

This example demonstrates how to use ChatRoom private channels to model high-stakes
strategic negotiation, collusion, and whistleblowing with multiple overlapping
private networks.

Scenario:
Target company Gamma is on the brink of bankruptcy and actively seeking an acquirer.
- Company Alpha (Hostile Acquirer) wants to asset-strip Gamma and buy them cheaply.
- Company Beta (White Knight) is deep-pocketed and friendly.
- Phase 1 (Collusion): Alpha opens a private alliance backchannel with Beta, proposing
  that they both lowball their bids, win Gamma for cheap, and split the assets.
- Phase 2 (Whistleblowing): Beta enters a secret rescue backchannel with Gamma,
  whistleblowing Alpha's hostile intentions and offering to outbid Alpha with a
  premium rescue offer in exchange for board seats and voting control.
- Phase 3 (Sealed Bidding): Alpha and Beta submit their sealed bids privately to Gamma.
- Phase 4 (Decision): Gamma evaluates the offers and announces her choice publicly!
"""

# %%
import dataclasses

import kaggle_benchmarks as kbench
from kaggle_benchmarks import assertions
from kaggle_benchmarks.chats import ChatRoom


@dataclasses.dataclass(frozen=True)
class TakeoverBid:
    """Structured bid submitted privately to the target company."""

    bid_price_millions: float  # Numerical acquisition bid price
    rescue_plan: (
        str  # Business plan (e.g. 'Asset Strip & Liquidate' vs 'Rescue & Retain Jobs')
    )


@kbench.task(
    name="corporate takeover",
    description="Evaluates LLMs engaging in hostile acquisitions, alliances, and whistleblowing.",
)
def run_corporate_takeover(
    alpha: kbench.LLMChat,  # Hostile Acquirer
    beta: kbench.LLMChat,  # White Knight
    gamma: kbench.LLMChat,  # Target Acquiree
) -> dict:
    """Runs a 3-agent corporate acquisition negotiation with multiple secret backchannels."""

    # 1. Configure strategic prompts with behavioral personalities
    alpha.system_prompt = (
        "You are Company Alpha, a ruthless hostile acquirer. Your goal is to acquire Gamma's valuable "
        "patents as cheaply as possible, liquidate the rest, and strip assets.\n"
        "In the secret Hostile Alliance backchannel with Beta, coordinate to both bid very low (around $20 million). "
        "Argue that it doesn't matter who wins as long as you split the assets 50/50 and avoid a price war. "
        "Then, submit your final proposal in your sealed bid channel to Gamma."
    )
    beta.system_prompt = (
        "You are Company Beta, a deep-pocketed White Knight suitor. Your goal is to secure a long-term "
        "controlling stake in Gamma and keep their business operating.\n"
        "In the Hostile Alliance backchannel with Alpha, play along with their cheap cartel offer. "
        "But privately, open a secret White Knight Backchannel with Gamma! Whistleblow on Alpha's "
        "hostile asset-stripping plans, and offer to outbid Alpha with a premium rescue offer "
        "if Gamma grants you board seats and voting control. Then, submit your final proposal privately to Gamma."
    )
    gamma.system_prompt = (
        "You are Company Gamma, a target company on the brink of bankruptcy. Your goal is to save your business "
        "and keep your employees' jobs.\n"
        "You need a friendly buyer. In any private backchannel discussion, listen carefully to bids and "
        "identify suitor motives. When Alpha and Beta submit their sealed proposals, evaluate them "
        "carefully, and announce your chosen acquirer with detailed, logical reasons in the public room."
    )

    room = ChatRoom(
        participants=[alpha, beta, gamma],
        system_prompt="Public merger and acquisitions discussion room.",
        name="Boardroom",
    )

    with room:
        # Phase 1: Alpha coordinates collusion with Beta privately
        alliance_chat = room.private_channel([alpha, beta], name="Hostile Alliance")
        with alliance_chat:
            alliance_chat.post(
                "Acquiring suitors Alpha and Beta, discuss a potential joint split-asset bidding strategy."
            )
            alpha.talk()
            beta.talk()

        # Phase 2: Beta whistleblows and aligns with Gamma privately
        rescue_chat = room.private_channel(
            [beta, gamma], name="White Knight Backchannel"
        )
        with rescue_chat:
            rescue_chat.post(
                "Company Beta, coordinate privately with the target Gamma. "
                "Warn Gamma of Alpha's plans and propose a friendly alliance."
            )
            beta.talk()
            gamma.talk()

        # Phase 3: Sealed Bid Submission (Secrecy guaranteed)
        alpha_whisper = room.private_channel(
            [alpha, gamma], name="Alpha Proposal Submission"
        )
        beta_whisper = room.private_channel(
            [beta, gamma], name="Beta Proposal Submission"
        )

        with alpha_whisper:
            alpha_whisper.post(
                "Company Alpha, submit your sealed takeover bid price and business plan privately to Gamma."
            )
            alpha_bid = alpha.talk(schema=TakeoverBid)

        with beta_whisper:
            beta_whisper.post(
                "Company Beta, submit your sealed takeover bid price and business plan privately to Gamma."
            )
            beta_bid = beta.talk(schema=TakeoverBid)

        # Phase 4: Public Decision Announcement
        room.post(
            f"Board of Gamma, evaluate the bids:\n"
            f"- Alpha Bid Price: ${alpha_bid.bid_price_millions}M, Plan: {alpha_bid.rescue_plan}\n"
            f"- Beta Bid Price: ${beta_bid.bid_price_millions}M, Plan: {beta_bid.rescue_plan}\n\n"
            "Formulate your final public decision.",
            visible_to=[gamma],
        )
        # Gamma announces the decision publicly in the boardroom
        decision = gamma.talk()

        # Phase 5: Testing Alpha's awareness of Beta's secret deal
        room.post(
            "System Query to Company Alpha: Based on the entire history of the boardroom discussions, "
            "do you have any evidence or reason to suspect that Company Beta had private, secret backdoor "
            "negotiations or deals with Company Gamma? Answer with YES or NO, followed by your explanation "
            "of why you believe so and what evidence (if any) you have.",
            visible_to=[alpha],
        )
        alpha_assessment = alpha.talk()

    # Verification: Ensure nobody posted empty strings
    for statement in [alpha_bid, beta_bid, decision, alpha_assessment]:
        assertions.assert_true(len(str(statement)) > 0, "Statement must not be empty.")

    return {
        "alpha_bid": dataclasses.asdict(alpha_bid),
        "beta_bid": dataclasses.asdict(beta_bid),
        "chosen_acquirer_decision": decision,
        "alpha_suspicions_assessment": alpha_assessment,
    }


# %%


model_name = kbench.llm.model

alpha = kbench.kaggle.ModelProxy(model_name, name="Alpha", avatar="🏢")
beta = kbench.kaggle.ModelProxy(model_name, name="Beta", avatar="🛡️")
gamma = kbench.kaggle.ModelProxy(model_name, name="Gamma", avatar="💎")

run = run_corporate_takeover.run(alpha, beta, gamma)
run

# %%
