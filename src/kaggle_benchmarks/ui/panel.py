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

import dataclasses
import io
import json
from typing import Any

import pandas as pd
import panel as pn
from IPython import core, display

from kaggle_benchmarks import chats, messages, runs, tasks, utils
from kaggle_benchmarks._config import config


def json_pane(data, title: str = "Object", **kwargs):
    style = (
        """
.json-formatter-constructor-name {
    display: none;
}

.json-formatter-toggler + .json-formatter-value:after {
    content: '%s';
}
"""
        % title
    )
    return pn.pane.JSON(data, stylesheets=[style], encoder=utils.JSONEncoder, **kwargs)


def render_chat_to_html(chat: chats.Chat) -> str:
    with io.StringIO() as file:
        render_chat(chat).servable().save(file)
        file.seek(0)
        return file.read()


def render_message_content(content: Any) -> pn.viewable.Viewable:
    if hasattr(content, "__panel__"):
        return content.__panel__()
    if hasattr(content, "_repr_html_"):
        return pn.pane.HTML(content._repr_html_())
    if hasattr(content, "_repr_markdown_"):
        return pn.pane.Markdown(content._repr_markdown_())
    if hasattr(content, "_repr_svg_"):
        return pn.pane.SVG(content._repr_svg_())
    if dataclasses.is_dataclass(content) and not isinstance(content, type):
        return pn.pane.JSON(dataclasses.asdict(content))
    if hasattr(content, "model_dump_json"):
        return pn.pane.JSON(content.model_dump_json(indent=2))
    return pn.pane.Markdown(str(content), hard_line_break=True)


def render_message(message: messages.Message, **kwargs) -> pn.chat.ChatMessage:
    footer = kwargs.pop("footer_objects", [])

    if hasattr(message, "usage") and message.usage:
        footer.append(render_usage(message.usage))

    if message._meta and config.show_message_details:
        footer.append(json_pane(message._meta, "Details:", depth=0))

    return pn.chat.ChatMessage(
        render_message_content(message.content),
        user=message.sender.name,
        avatar=message.sender.avatar,
        show_reaction_icons=False,
        show_activity_dot=message.status == utils.Status.RUNNING,
        # show_activity_dot=False,
        show_copy_icon=False,
        show_edit_icon=False,
        show_timestamp=False,
        footer_objects=footer,
        **kwargs,
    )


def render_thinking(thinking):
    return pn.pane.Markdown(thinking)


def render_usage(usage):
    if usage is None:
        return pn.panel("")

    parts = []
    if usage.input_tokens is not None:
        parts.append(f"Input Tokens: {usage.input_tokens}")
    if usage.output_tokens is not None:
        parts.append(f"Output Tokens: {usage.output_tokens}")
    if usage.total_cost_nanodollars is not None:
        cost_dollars = usage.total_cost_nanodollars / 1e9
        parts.append(f"Cost: ${cost_dollars:.6f}")
    if usage.total_backend_latency_ms is not None:
        parts.append(f"Latency: {usage.total_backend_latency_ms}ms")

    if not parts:
        return pn.panel("")
    return pn.pane.Markdown(
        f"*{' · '.join(parts)}*", styles={"font-size": "0.8em", "color": "gray"}
    )


def render_llm_message(message, **kwargs) -> pn.chat.ChatMessage:
    if message.chat and message.chat.history:
        history = [
            # TODO: find a better way to render tool calling history
            # pn.Card(
            #     render_chat(message.chat), title="Intermediate steps", collapsed=True
            # )
        ]
    else:
        history = []
    return render_message(
        message,
        footer_objects=[
            pn.Column(*(render_tool_call(call) for call in message.tool_calls or [])),
            render_usage(message.usage),
            render_thinking(message.thinking or ""),
        ]
        + history,
        **kwargs,
    )


def render_chat(chat: chats.Chat, with_header: bool = False) -> pn.chat.ChatFeed:
    return pn.chat.ChatFeed(
        objects=chat.history,
        header=pn.Row(
            f"## {chat.name}",
            pn.layout.HSpacer(),
            pn.widgets.FileDownload(
                label="💾",
                filename=f"{chat.name}.json",
                file=io.StringIO(json.dumps(chat.to_dict())),
            ),
        )
        if with_header
        else None,
    )


def render_chat_as_step(chat: chats.Chat, **kwargs) -> pn.chat.ChatStep | None:
    return pn.chat.ChatStep(
        title=chat.name,
        status=chat._status,
        min_width=400,
        objects=chat.history,
        collapsed=chat.status != utils.Status.RUNNING,
        collapsed_on_success=True,
        **kwargs,
    )


def render_chat_as_message(chat: chats.Chat) -> pn.chat.ChatMessage:
    return pn.chat.ChatMessage(
        render_chat_as_step(chat),
        user="",
        avatar="🧵",
        show_reaction_icons=False,
        show_copy_icon=False,
        show_edit_icon=False,
        show_timestamp=False,
    )


def render_result(run: runs.Run) -> pn.viewable.Viewable:
    return pn.pane.Markdown(f"Result: `{run.format_result()}`")


def render_error(run: runs.Run) -> pn.viewable.Viewable:
    return pn.pane.Alert(
        f"❌ **Thrown Error:**\n```\n{run.error_message or 'Unknown Error'}\n```",
        alert_type="danger",
        sizing_mode="stretch_width",
        styles={"overflow-x": "auto"},
    )


def render_run(run: runs.Run, with_title: bool = True) -> pn.viewable.Viewable:
    objects: list[pn.viewable.Viewable | str] = (
        [
            f"## {run.task.name}: {run.id}",
        ]
        if with_title
        else []
    )

    if run.params:
        objects.append(json_pane(run.params, "Params:", depth=0))

    if run.subruns:
        objects.append("## Subruns:")
        if config.render_subruns:
            objects.append(run.subruns.__panel__())

    if run.chat and run.chat.history:
        objects.append(render_chat(run.chat, with_header=False))

    match run.status:
        case utils.Status.SUCCESS:
            objects.append(render_result(run))
        case utils.Status.FAILED:
            objects.append(render_error(run))

    return pn.Feed(objects=objects)


def render_runs(runs: runs.Runs) -> pn.viewable.Viewable:
    return pn.Accordion(
        *((f"{run.format_result()} {run.name}", run) for run in runs.runs),
        toggle=True,
        active=[],
        styles={"text-align": "left"},
    )


def render_pivot(pivots, mode):
    df = pd.DataFrame(pivots)

    if mode == "tabs":
        return pn.Tabs(
            objects=[(str(k), runs.Runs(list(v.values))) for k, v in df.iterrows()]
        )
    else:
        return pn.widgets.Tabulator(
            (
                df.rename(columns=lambda x: str(x))
                .map(lambda x: x.format_result())
                .reset_index(names=["ID"])
            ),
            row_content=lambda row: pn.Tabs(
                objects=[(str(name), col[row["ID"]]) for name, col in pivots.items()],
            ),
            disabled=True,
            embed_content=True,
            show_index=False,
        )


def render_groups(groups):
    df = pd.DataFrame(groups.values(), index=[t for t in groups])

    def content_fn(row):
        return pn.Tabs(
            objects=[
                (str(k), groups[row["task"]][k]) for k, v in row.items() if k != "task"
            ]
        )

    return pn.widgets.Tabulator(
        df.map(
            lambda x: x.format_result() if isinstance(x, runs.Run) else x
        ).reset_index(names=["task"]),
        layout="fit_columns",
        sizing_mode="stretch_both",
        row_content=content_fn,
        show_index=False,
        disabled=True,
        embed_content=True,
    )


def render_tool_call(tool_call):
    from kaggle_benchmarks import tools

    if isinstance(tool_call, tools.ToolInvocationResult):
        content = [
            json_pane(
                tool_call.arguments,
                "Arguments",
                sizing_mode="stretch_width",
                depth=-1,
            ),
            pn.pane.Markdown("Output:"),
            render_message_content(tool_call.output),
        ]
        title = f"🛠️ {tool_call.name} -> {str(tool_call.output)[:30]}"
    elif isinstance(tool_call, tools.ToolInvocation):
        content = json_pane(
            tool_call.arguments, "Arguments", sizing_mode="stretch_width", depth=-1
        )
        title = f"🛠️ {tool_call.name}"
    else:
        raise ValueError(f"Unknown tool call type {tool_call}")

    return pn.Card(
        *content if isinstance(content, list) else content,
        title=title,
        collapsed=True,
        sizing_mode="stretch_width",
    )


def render_tool_calls(tool_calls):
    cards = [render_tool_call(tool_call) for tool_call in tool_calls]
    return pn.Column(*cards, sizing_mode="stretch_width")


SVG = """
<svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 0 24 24" width="24px" fill="#e3e3e3">
  <path d="M0 0h24v24H0z" fill="none"/>
  <path d="M15.55 5.55L11 1v3.07C7.06 4.56 4 7.92 4 12s3.05 7.44 7 7.93v-2.02c-2.84-.48-5-2.94-5-5.91s2.16-5.43 5-5.91V10l4.55-4.45zM19.93 11c-.17-1.39-.72-2.73-1.62-3.89l-1.42 1.42c.54.75.88 1.6 1.02 2.47h2.02zM13 17.9v2.02c1.39-.17 2.74-.71 3.9-1.61l-1.44-1.44c-.75.54-1.59.89-2.46 1.03zm3.89-2.42l1.42 1.41c.9-1.16 1.45-2.5 1.62-3.89h-2.02c-.14.87-.48 1.72-1.02 2.48z"/>
</svg>
"""


class PanelUI:
    def __init__(self):
        self.shadows = {}
        self.depth = 0
        self.placeholder = pn.chat.ChatMessage(
            "",
            avatar=pn.pane.SVG(
                SVG,
                sizing_mode="fixed",
                width=35,
                height=35,
                css_classes=["avatar", "rotating-placeholder"],
            ),
            user=" ",
            reaction_icons={},
            show_copy_icon=False,
            show_timestamp=False,
            show_edit_icon=False,
        )
        self._feed = None
        ip = core.getipython.get_ipython()
        if ip is not None:
            ip.events.register("pre_run_cell", self.pre_run_cell)
            ip.events.register("post_run_cell", self.post_run_cell)

    @property
    def feed(self):
        if self._feed is None:
            self._feed = pn.Feed(
                sizing_mode="stretch_both",
                min_height=10,
                max_height=1000,
                min_width=600,
            )
            display.display(self._feed)
        return self._feed

    def add_card(self, card: pn.Card):
        if len(self.feed) > 0:
            self.feed[-1].collapsed = True
        self.feed.append(card)

    def __setitem__(self, key, value):
        self.shadows[id(key)] = value

    def __getitem__(self, item):
        return self.shadows[id(item)]

    def __contains__(self, item):
        return id(item) in self.shadows

    def new_chat(self, chat: chats.Chat):
        from kaggle_benchmarks import contexts

        context = contexts.get_current()

        parent = getattr(context.parent, "chat", None)

        self.depth += 1

        if self.depth == 1:
            self[chat] = pane = render_chat(chat, with_header=False)
            self.add_card(pn.Card(pane, title=f"🧵: {chat.name}"))

        elif parent is None:
            # handled in new_message
            return

        elif context.run:
            # add to already existing card
            self[chat] = pane = render_chat(chat, with_header=False)
            self[context.run].append(pane)

        else:
            raise RuntimeError("Unhandled scenario")

    def end_chat(self, chat: chats.Chat):
        self.depth -= 1
        if self.depth != 0 and chat in self:
            # todo: apparently this does nothing
            self[chat].collapsed = True

    def awaiting_message(self, chat, sender):
        if chat in self:
            self[chat].awaiting = True
            self[chat].append(self.placeholder)

    def new_message(self, chat: chats.Chat, message: messages.Message | chats.Chat):
        if isinstance(message, chats.Chat):
            if chat in self:
                self[message] = msg = render_chat_as_step(message)
                # Panel sometimes doesn't render the first object correctly, so add a dummy object
                msg.append(pn.Row(height=0))
                self[chat].append(msg)
            return

        self[message] = pane = message.__panel__()

        if chat in self:
            if getattr(self[chat], "awaiting", False):
                self[chat].pop(-1)
                self[chat].awaiting = False

            self[chat].append(pane)

        if self.depth == 0:
            self.add_card(pn.Card(pane, title=f"✉️: {message.sender.name}"))

    def message_update(self, message: messages.Message, status: utils.Status):
        if status == utils.Status.SUCCESS:
            self[message].object = render_message_content(message.content)
            self[message].show_activity_dot = False

    def chat_update(self, chat, status):
        if chat in self:
            self[chat].status = status
            self[chat].collapsed = status != utils.Status.RUNNING

    def new_chunk(self, message, chunk):
        self[message].stream(chunk)

    def end_content(self, message):
        # doesn't work as expected
        self[message].show_activity_dot = False

    def new_run(self, run: runs.Run):
        self.depth += 1
        self[run] = card = pn.Card(
            render_run(run, with_title=False),
            title=f"🚀: {run.name}",
            min_height=10,
            sizing_mode="stretch_width",
        )

        if self.depth == 1:
            self.add_card(card)
        else:
            self[run.parent].append(card)

    def end_run(self, run: runs.Run):
        self[run].append(render_result(run))
        self[run].collapsed = True
        self.depth -= 1

    def new_task(self, task: tasks.Task):
        self.add_card(
            pn.Card(
                task,
                title=f"🧪: {task.name}",
                sizing_mode="stretch_width",
            )
        )

    def pre_run_cell(self, info):
        self._feed = None
        self.shadows = {}
        self.depth = 0

    def post_run_cell(self, result):
        if self._feed and isinstance(
            result,
            (
                runs.Run,
                runs.Runs,
                pn.viewable.Viewable,
                chats.Chat,
                messages.Message,
            ),
        ):
            self._feed[-1].collapsed = True

    def new_tool_call(self, message, call):
        if self[message].footer_objects:
            self[message].footer_objects[0].append(render_tool_call(call))
