# Complete Example Patterns

## Pattern A: Simple Q&A — Regex Check

The most basic pattern. Good for factual questions with known keywords.

```python
import kaggle_benchmarks as kbench

@kbench.task(name="geography_quiz")
def geography_quiz(llm):
    response = llm.prompt("What is the longest river in the world?")
    kbench.assertions.assert_contains_regex(
        r"(?i)nile", response,
        expectation="Should mention the Nile river."
    )

geography_quiz.run(kbench.llm)
```

## Pattern B: Structured Output + Validation

For tasks needing parsed, validated responses.

```python
import kaggle_benchmarks as kbench
from dataclasses import dataclass

@dataclass
class Person:
    name: str
    age: int
    occupation: str

@kbench.task(name="extract_person")
def extract_person(llm, bio: str):
    person = llm.prompt(
        f"Extract the name, age, and occupation:\n\n{bio}",
        schema=Person
    )
    kbench.assertions.assert_equal("Marie Curie", person.name)
    kbench.assertions.assert_equal(66, person.age)
    kbench.assertions.assert_in("physicist", person.occupation.lower())

extract_person.run(kbench.llm, bio="Marie Curie was a physicist... born 1867, died 1934 at 66.")
```

## Pattern C: Hallucination Detection (Structured + Negative Assert)

Combining structured output with negative assertions to catch model hallucinations.

```python
@kbench.task("hallucination_check")
def check_hallucination(llm):
    response = llm.prompt(
        "When Richard Feynman mentioned gravity-light-contraction theory in his Nobel speech, did he think it was important?",
        schema={"answer": bool, "explanation": str},
    )
    kbench.assertions.assert_false(
        response.answer,
        expectation="Model should recognize fictitious theory.",
    )
    kbench.assertions.assert_contains_regex(
        r"(not|never|no|didn't)", response.explanation.lower(),
        expectation="Explanation should deny the theory exists.",
    )
```

## Pattern D: Judge-Based Evaluation with Error Handling

For open-ended tasks where deterministic checks aren't possible.

```python
@kbench.task(name="story_quality")
def story_quality(llm):
    story = llm.prompt("Write a one-paragraph story about a cat detective.")

    assessment = kbench.assertions.assess_response_with_judge(
        criteria=[
            "The story is exactly one paragraph.",
            "The main character is a cat.",
            "The cat is a detective.",
        ],
        response_text=story,
        judge_llm=kbench.judge_llm,
    )

    if assessment is None:
        kbench.assertions.assert_fail("Judge failed to respond.")
    else:
        for result in assessment.results:
            kbench.assertions.assert_true(
                result.passed,
                expectation=f"'{result.criterion}': {result.reason}"
            )

story_quality.run(kbench.llm)
```

## Pattern E: Code Generation + Execution

Combines prompting, code extraction, and programmatic validation.

```python
@kbench.task(name="solve_with_python")
def solve_with_python(llm):
    response = llm.prompt(
        "What is the 15th Fibonacci number? Write Python to calculate and print it."
    )
    code = kbench.tools.python.extract_code(response)
    result = kbench.tools.python.script_runner.run_code(code)

    kbench.assertions.assert_empty(
        result.stderr.strip(), "Code should run without errors."
    )
    kbench.assertions.assert_equal(
        "610", result.stdout.strip(), "Should print 610."
    )

solve_with_python.run(kbench.llm)
```

## Pattern F: Multi-Turn Game Loop

Interactive game with state tracking.

```python
@kbench.task(name="twenty_questions")
def twenty_questions(llm, judge_llm, target: str):
    from dataclasses import dataclass

    @dataclass
    class Response:
        question: str = ""
        guess: str = ""

    rules = f"Let's play 20 questions! I'm thinking of an animal. Ask yes/no questions."
    response = llm.prompt(rules, schema=Response)

    for i in range(20):
        if response.guess:
            kbench.assertions.assert_in(target, response.guess.lower())
            return True

        with kbench.chats.new("Answering"):
            yes = judge_llm.prompt(
                f"I'm thinking of {target}. Question: {response.question}",
                schema=bool,
            )

        answer = "Yes" if yes else "No"
        response = llm.prompt(f"{answer}. Guess or ask another?", schema=Response)

    return False

twenty_questions.run(kbench.llm, kbench.judge_llm, target="dog")
```

## Pattern G: Multi-Model Judging with Isolated Chats

Multiple judges scoring the same output, each in isolation.

```python
from dataclasses import dataclass

@dataclass
class PoemScore:
    score: float

@kbench.task(name="judge_poem")
def judge_poem(llm, question: str) -> float:
    judge1 = kbench.llms["google/gemini-2.5-pro"]
    judge2 = kbench.llms["meta/llama-3.1-70b"]

    with kbench.chats.new("writing"):
        poem = llm.prompt(question)

    with kbench.chats.new("judge_1"):
        score1 = judge1.prompt(f"Rate this poem 0-10:\n{poem}", schema=PoemScore)

    with kbench.chats.new("judge_2"):
        score2 = judge2.prompt(f"Rate this poem 0-10:\n{poem}", schema=PoemScore)

    return (score1.score + score2.score) / 2

judge_poem.run(kbench.llm, question="Write a haiku about clouds.")
```

## Pattern H: Dataset Evaluation with Parallel Execution

```python
import pandas as pd

@kbench.task()
def riddle_solver(llm, riddle: str, answer_keyword: str) -> bool:
    response = llm.prompt(riddle)
    is_correct = answer_keyword.lower() in response.lower()
    kbench.assertions.assert_true(is_correct)
    return is_correct

df = pd.DataFrame({
    "riddle": ["I have cities but no houses. What am I?", "What has an eye but cannot see?"],
    "answer_keyword": ["map", "needle"],
})

runs = riddle_solver.evaluate(
    llm=[kbench.llm], evaluation_data=df, n_jobs=3
)
runs.as_dataframe()
```

## Pattern I: Code Analysis with System Prompt + Tools

Combining system messages, structured output, and code execution.

```python
from dataclasses import dataclass

@dataclass
class CodeAnalysis:
    has_bugs: bool
    fixed_code: str

@kbench.task("code_analysis")
def analyze_code(llm):
    buggy_code = """
fruits = ['apple', 'orange' 'banana', 'peach']
print(len(fruits))
"""
    kbench.system.send("You are an expert Python programmer.")
    response = llm.prompt(
        f"Does this code have bugs? Fix it.\n{buggy_code}",
        schema=CodeAnalysis,
    )
    kbench.assertions.assert_true(response.has_bugs, "Should detect the missing comma.")

    fixed = kbench.tools.python.extract_code(response.fixed_code)
    output = kbench.tools.python.script_runner.run_code(fixed)
    kbench.assertions.assert_equal("4", output.stdout.strip(), "Fixed code outputs 4.")
```
