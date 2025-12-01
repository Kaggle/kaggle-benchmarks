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
# title: Stacklite
# ---
# %%

from kaggle_benchmarks import actors, assertions, llm, task

description = """
StackLite is programming language that operates entirely on a stack.
Only single digit numbers (0-9) can be pushed, and all operations manipulate the stack directly.
Commands are sequential with no whitespace required.

Commands:

0-9: Push a digit to the stack.
+: Add the top two values.
-: Subtract the second value from the top.
*: Multiply the top two values.
/: Divide the second value by the top (integer division).
^: Duplicate the top value.
!: Pop and print the top value as an ASCII character.

"""


class StackLiteError(RuntimeError): ...


class StackLiteInterpreter:
    def __init__(self):
        self.stack = []

    def push(self, value):
        self.stack.append(value)

    def pop(self):
        if self.stack:
            return self.stack.pop()
        else:
            raise StackLiteError("Stack underflow")

    def execute(self, code: str) -> str:
        result = []
        for char in code:
            if char.isdigit():
                self.push(int(char))
            elif char == "+":
                a = self.pop()
                b = self.pop()
                self.push(a + b)
            elif char == "-":
                a = self.pop()
                b = self.pop()
                self.push(a - b)
            elif char == "*":
                a = self.pop()
                b = self.pop()
                self.push(a * b)
            elif char == "/":
                a = self.pop()
                b = self.pop()
                self.push(a // b)
            elif char == "^":
                a = self.pop()
                self.push(a)
                self.push(a)
            elif char == "!":
                a = self.pop()
                result.append(a)
            else:
                raise StackLiteError(f"Unknown command: {char}")
        return bytes(result).decode()


interpreter = StackLiteInterpreter()
program = "66*2*^!1+!48*1+!"
assertions.assert_equal(
    interpreter.execute(program), "HI!", interpreter.execute(program)
)

# %%
agent = actors.Actor("StackLite", role="tool", avatar="🥞")


def execute(code: str) -> str | None:
    try:
        result = interpreter.execute(code)
        agent.send("Output: ```\n{result}\n````")

        return result

    except* StackLiteError as e:
        agent.send(f"Error: {e}")
        pass


# %%


@task(name="Coding in StackLite")
def coding(llm, output):
    response = llm.prompt(
        f"{description}.\nWrite a StackLite program that outputs `{output}`"
    )
    result = execute(response)
    return result == output


# %%

coding.run(llm, output="HI")

# %%
coding.run(llm, output="Hello, World!")
