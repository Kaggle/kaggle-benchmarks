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
# title: Doom
# ---

# %%
# | code-fold: true

import math
import os

import moviepy
import numpy as np
import vizdoom as vzd
from PIL import Image

from kaggle_benchmarks import (
    actors,
    assertions,
    chats,
    content_types,
    llm,
    messages,
    task,
)


def get_available_scenarios():
    path = vzd.scenarios_path
    return [s[: -len(".cfg")] for s in os.listdir(path) if s.endswith(".cfg")]


START_PROMPT = """
You will be playing a game of DOOM.
Your goal is to select actions and defeat Monsters.
Available actions: {actions}
You can select exactly one action per turn.

Think step by step before making a decision.
Your ATTACK only affects enemies perfectly in front of you.
If your enemies are on your left, you need to move LEFT before you ATTACK.
If your enemies are on your right, you need to move RIGHT before you ATTACK.
Don't ATTACK too often. In particular, do not ATTACK twice in a row.
Wasting ammunition might result in a negative reward.
A single successful ATTACK is enough to defeat a Monster.
After a successful ATTACK the Monster will disappear IMMEDIATELY
and you will get a positive reward.

Analyze effects of the previous move after every turn.
In particular, think whether your attack missed or hit.
Also, think whether your target is on the LEFT or on the RIGHT.
If you missed an attack, try to learn from this mistake and reposition.
You need to move multiple turns in a row before you get in a position to attack.

The game doesn't stop when you kill the monster. Continue producing the commands.
After thinking, select a single action and display its full name.
"""


class DoomGameWrapper:
    def __init__(self, scenario, seed):
        self.scenario = scenario
        self.frames = []
        self.seed = seed
        self.total_steps = 0
        self.total_reward = 0
        self.total_rounds = 0

        self.reset()

    def reset(self):
        game = vzd.DoomGame()

        # ViZDoom must create a temporary file with configuration
        game.set_doom_config_path("/tmp/_vizdoom.ini")

        game.set_seed(self.seed + self.total_rounds)
        self.total_rounds += 1

        game.load_config(self.scenario)
        game.set_screen_format(vzd.ScreenFormat.RGB24)

        game.set_render_hud(True)
        game.set_render_crosshair(True)
        game.set_render_weapon(True)
        game.set_render_decals(False)  # Bullet holes and blood on the walls
        game.set_render_particles(False)
        game.set_render_effects_sprites(False)  # Like smoke and blood
        game.set_render_messages(False)  # In-game text messages
        game.set_render_corpses(False)
        game.set_render_screen_flashes(False)  # When taking damage or picking up items

        game.set_living_reward(0)
        game.set_window_visible(False)
        game.set_mode(vzd.Mode.PLAYER)
        game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
        game.init()

        self.game: vzd.DoomGame = game
        frame = self.game.get_state().screen_buffer
        self.frames.append(frame)
        return frame

    @property
    def available_actions(self):
        return [button.name for button in self.game.get_available_buttons()]

    def repeat_action(self, selected_action, repeat):
        if isinstance(selected_action, str):
            selected_action = self.available_actions.index(selected_action)

        rewards = 0
        action = np.full([repeat, len(self.available_actions)], fill_value=False)

        if self.available_actions[selected_action] == "ATTACK":
            # We press the trigger only once
            # at the beginning of the frame stack
            action[0, selected_action] = True
        if "TURN" in self.available_actions[selected_action]:
            # Slower turning
            half_turn = math.ceil(repeat / 2)
            action[:half_turn, selected_action] = True
        else:
            action[:, selected_action] = True

        for row in action:
            rewards += self.game.make_action(row)
            if self.game.get_state() is None:
                return rewards

            frame = self.game.get_state().screen_buffer
            if frame is not None:
                self.frames.append(frame)
        return rewards

    def step(self, action: int | str, skipframe: int):
        reward = self.repeat_action(action, skipframe)
        self.total_reward += reward
        self.total_steps += 1
        return self.frames[-1], reward

    def display(self):
        return Image.fromarray(self.frames[-1])

    def display_video(self):
        video = moviepy.ImageSequenceClip(self.frames, fps=35)
        return moviepy.display_in_notebook(
            video, verbose=False, rd_kwargs=dict(logger=None)
        )

    def is_finished(self):
        return self.game.is_episode_finished()

    def get_results(self):
        return {
            "num_steps": self.total_steps,
            "total_reward": self.total_reward,
        }

    def repeat_last_frame(self, n: int):
        """Add last frame multiple times."""
        self.frames += [self.frames[-1]] * n


class Doom(actors.Actor):
    def __init__(self, scenario="basic", seed=0):
        super().__init__(name="Doom", role="user", avatar="🎮")
        self.game_wrapper = DoomGameWrapper(scenario=scenario, seed=seed)
        self.skipframe = 7

    def parse_action(self, response) -> str:
        assert isinstance(response, str)

        for word in response.split()[::-1]:
            for action in self.game_wrapper.available_actions:
                if action in word:
                    return action

        raise ValueError(f"Could not parse action from '{response.strip()}'")

    def initialize(self, start_prompt=None):
        # Send initial prompt to the chat
        if start_prompt is None:
            start_prompt = START_PROMPT.format(
                actions=self.game_wrapper.available_actions
            )
        self._send(start_prompt, is_visible_to_llm=True)

        # Send the latest RGB frame to the chat
        obs = self.game_wrapper.frames[-1]
        self._send(content_types.images.from_array(obs), is_visible_to_llm=True)

    def respond(self):
        chat = chats.get_current_chat()
        return self.send(chat.messages[-1])

    def send(self, message: str | messages.Message) -> messages.Message[str]:
        """Extract the Action and send the new observation to the chat."""
        if isinstance(message, messages.Message):
            message = message.text

        if self.game_wrapper.is_finished():
            self.game_wrapper.repeat_last_frame(n=14)
            obs = self.game_wrapper.reset()
            reward = None
        else:
            action = self.parse_action(message)
            obs, reward = self.game_wrapper.step(
                action,
                skipframe=self.skipframe,
            )

        self._send(content_types.images.from_array(obs), is_visible_to_llm=True)

        if reward is not None:
            return self._send(f"LAST MOVE REWARD: {reward}", is_visible_to_llm=True)
        else:
            return self._send(
                "ROUND FINISHED! NEW ROUND STARTED!", is_visible_to_llm=True
            )

    def display_video(self):
        return self.game_wrapper.display_video()


# %%


@task(name="Can it play DOOM?")
def can_it_play_doom(llm):
    """
    Test if your LLM can play the game of DOOM!
    Pass the test by defeating a monster.
    """
    doom = Doom(scenario="basic", seed=0)

    with chats.new("DOOM"):
        # Sends the game instructions to the agent
        # Shows the first frame to the agent
        doom.initialize()

        for turn in range(20):
            llm.respond()  # LLM selects an action
            doom.respond()  # Parse agent response and show the next frame

    assertions.assert_true(
        doom.game_wrapper.total_reward > 0,
        f"LLM should win the game with a positive reward. LLM scored {doom.game_wrapper.total_reward}",
    )


can_it_play_doom.run(llm)

# %%
