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

from kaggle_benchmarks import actors, chats, ui


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
