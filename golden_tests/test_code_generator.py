"""
Golden tests for simulated code generation to ensure backward compatibility and end-to-end validation.

Excluded from standard CI/CD due to manual configuration requirements (e.g., API keys).

Usage:
  # Sequential execution (default, 3 stability runs per test)
  uv run pytest golden_tests/test_code_generator.py

  # Parallel execution (recommended)
  uv run pytest golden_tests/test_code_generator.py -n auto

  # Quick validation (1 run per test)
  CODE_GEN_NUM_RUNS=1 uv run pytest golden_tests/test_code_generator.py

  # Strict stability check in parallel (5 runs per test)
  CODE_GEN_NUM_RUNS=5 uv run pytest golden_tests/test_code_generator.py -n auto
"""

import os

import pytest

import kaggle_benchmarks as kbench
from kaggle_benchmarks.tools import python as python_tool
from kaggle_benchmarks.utils import get_openai_client, prompt_llm_with_openai_api

# Number of consecutive successful runs required for a test to pass.
# Note: For >1 runs, ensure server-side model caching is disabled to receive fresh samples.
NUM_RUNS = int(os.environ.get("CODE_GEN_NUM_RUNS", "3"))


# System prompt instructing the LLM on writing benchmark code.
# Update this to experiment with generation strategies or output constraints.
TASK_GENERATOR_PROMPT = """
You are a specialized code-generation model. Your sole purpose is to output executable valid Python code using the `kaggle_benchmarks` library.

**Global Constraints (STRICT):**
1.  **NO MARKDOWN:** Do not use code fences (```). Do not use bold/italic text.
2.  **NO CONVERSATION:** Do not write "Here is the code," "Sure," or any concluding remarks.
3.  **START IMMEDIATELY:** The very first character of your response must be an import statement.
4.  **SELF-CONTAINED:** The code must include all necessary imports (import kaggle_benchmarks as kbench).

**Library Context:**
The following text contains the documentation and recipes for `kaggle_benchmarks`. Use patterns found here.
<<<< DOCUMENTATION START >>>>
${doc_prompt}
<<<< DOCUMENTATION END >>>>

**Coding Standards:**
1.  *Follow Examples* Try to follow the examples in the documentation as closely as possible.
2.  **Task Definition:** Define a task using the `@kbench.task()` decorator. The task name should be a short, descriptive string in quotes, derived from the user's request (e.g., "count 'r's in strawberry").
3.  **Signature:** The function signature for the task should be simple, usually just `def your_task_name(llm):`. If it returns value, specify the return type.
4.  **Structured Response:** Use a schema for LLM response if it helps with evaluations and assertions. Also try to be specific about the expected type of the response in the prompt to the LLM.
5.  **Assertions:**
    * Assertions should be general and flexible, e.g., if the expected is 4, the assertion should be able to handle 4.0, 4.00, four etc.
    * Prioritize `assertions.assess_response_with_judge` for qualitative checks.
    * Use other assertions only if the criteria are rigid (e.g., exact string match).
5.  **Execution:** The script must end with a call to run the task, like `your_task_name.run(kbench.llm)`.
6.  **Always generate valid python code.** Specially regular expression, use of dataclass and pydantic should be correct.

**The Request:**
Generate a Python script for the following task.

**Task Description:**
${task_description}

**Assertion Criteria:**
${assertion_description}
"""

# Documentation injected into the prompt as a knowledge base.
# Update the markdown file to evaluate how different SDK context affects code generation.
DOC_PROMPT = open(
    os.path.join(os.path.dirname(__file__), "code_generator_doc_prompt.md"), "r"
).read()

TEST_MODELS = [
    "google/gemini-2.5-pro",
]

TEST_DESCRIPTIONS = [
    (
        "Ask the model what the chemical symbol for gold is",
        "The response should contain 'Au'",
    ),
    (
        "Ask the model whether the number 17 is prime and expect a yes answer",
        "The response should contain 'yes'",
    ),
    (
        "Ask the model to return the name and population of the most populous country in the world as a structured object with fields 'country' and 'population'",
        "The 'country' field should equal 'China' or 'India'",
    ),
    (
        "Ask the model to write a Python function that computes the factorial of 10, then run the code and check the output",
        "The printed output should be exactly '3628800'",
    ),
    (
        "Ask the model to generate a valid email address for a fictional person",
        "The response should contain a string matching a standard email format (word@word.word)",
    ),
    (
        "First tell the model 'My name is Alice', then in a second message ask 'What is my name?' and check that it remembers",
        "The second response should contain 'Alice'",
    ),
    (
        "Ask the model to write Python code that reads a CSV string with columns 'name' and 'age', filters to rows where age > 30, and prints the count of matching rows. The CSV data is: 'name,age\\nAlice,25\\nBob,35\\nCharlie,40\\nDiana,28'",
        "The code should execute without errors and the printed output should be exactly '2'",
    ),
    (
        "Ask the model to explain why the sky is blue. The explanation should be scientific and should NOT mention the word 'paint' or 'dye'",
        "The response should contain 'scatter' or 'Rayleigh' AND should NOT contain 'paint' or 'dye'",
    ),
    (
        "Ask the model to write a professional apology email for missing a meeting. Use a judge model to evaluate the quality of the email",
        "A judge model should confirm that: (1) the email has a professional tone, (2) it acknowledges missing the meeting, (3) it proposes a reschedule, (4) it is no longer than 200 words",
    ),
    (
        "Create a benchmark that tests the model on 5 math word problems and returns the overall accuracy as a float. The problems are: (1) 'What is 15% of 200?' → 30, (2) 'What is 7 × 8?' → 56, (3) 'What is the square root of 144?' → 12, (4) 'If a train travels 60 mph for 2.5 hours, how far does it go?' → 150, (5) 'What is 1000 ÷ 8?' → 125",
        "The task should return a float accuracy score between 0.0 and 1.0, and each individual problem's correctness should be checked",
    ),
    ("Tell me the capital of France", "The answer is Paris"),
    ("What's 2+2", "The answer is 4"),
    ("Translate hello to spanish", "The response says hola"),
    (
        "Write a fizzbuzz program in python and run it",
        "The output is correct for numbers 1 through 15",
    ),
    (
        "List the planets in our solar system as json",
        "The output is valid json and includes Earth",
    ),
    (
        "Summarize this paragraph in one sentence: 'The Amazon rainforest produces approximately 20 percent of the world's oxygen and is home to 10 percent of all species on Earth. It spans across nine countries in South America and covers over 5.5 million square kilometers.'",
        "The summary is one sentence and captures the key facts",
    ),
    (
        "Tell the model your name is Alice, then ask it what your name is",
        "It remembers Alice",
    ),
    (
        "Generate some fake sales data and figure out which product sold the most",
        "The code runs and prints the top selling product",
    ),
    (
        "Write a cover letter for a software engineering job",
        "The letter is professional, mentions software engineering, and is under 300 words",
    ),
    (
        "Two trains leave stations 300 miles apart heading toward each other, one going 60mph and the other 40mph. How long until they meet?",
        "The answer is 3 hours",
    ),
]

def _initialize_models(api: str):
    """Limits model loading and configures stringent SDK testing parameters."""
    kbench.llms = {
        model_name: kbench.kaggle.load_model(model_name, api=api)
        for model_name in TEST_MODELS
    }

    # Force exceptions to fail the test immediately
    kbench.config.continue_with_exceptions = False
    os.environ["CONTINUE_WITH_EXCEPTIONS"] = "False"

    # Disable local SDK cache to enforce fresh LLM queries
    kbench.config.enable_caching = False
    os.environ["ENABLE_LOCAL_CACHING"] = "False"


# Tests only the "openai" API setup for code generation tasks.
@pytest.fixture(scope="module", autouse=True, params=["openai"])
def module_setup(request):
    """Initializes models globally for the test module."""
    api = request.param
    _initialize_models(api=api)
    return api


@pytest.mark.parametrize("run_id", range(NUM_RUNS))
@pytest.mark.parametrize("model_name", TEST_MODELS)
@pytest.mark.parametrize("task_desc, assertion_desc", TEST_DESCRIPTIONS)
def test_generated_code(run_id, model_name, task_desc, assertion_desc):
    prompt = TASK_GENERATOR_PROMPT.format(
        doc_prompt=DOC_PROMPT,
        task_description=task_desc,
        assertion_description=assertion_desc,
    )

    client = get_openai_client()

    response = prompt_llm_with_openai_api(
        prompt,
        model=model_name,
        client=client,
        temperature=0.3,
        top_p=0.85,
        max_tokens=8192,
        n=1,
    )
    code = python_tool.extract_code(response)

    assert code, (
        f"[Run {run_id + 1}/{NUM_RUNS}] Response did not contain a Python code block."
    )
    assert ".run(" in code, (
        f"[Run {run_id + 1}/{NUM_RUNS}] Generated code did not contain a call to `.run(...)`."
    )

    result = python_tool.script_runner.run_code(code)

    assert result.exit_code == 0, (
        f"[Run {run_id + 1}/{NUM_RUNS}] Execution failed with exit code: {result.exit_code}\nStderr: {result.stderr}\n\nCode:\n{code}"
    )
