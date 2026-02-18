# Local Development

Local Development is for internal team (kaggle) only. For local development, you will need to configure your environment to use the Kaggle Model Proxy.

**Prerequisites**:

- Python 3.11+
- Git
- [`uv`](https://docs.astral.sh/uv)

**Installation & Configuration**:

1.  Clone the repository:
    ```bash
    git clone https://github.com/Kaggle/kaggle-benchmarks.git
    cd kaggle-benchmarks
    ```
2.  Create a virtual environment and install dependencies using `uv`:

    ```bash
    # Create and activate the virtual environment
    uv venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`

    # Install dependencies
    uv pip install -e .
    ```

3.  Obtain a Kaggle `MODEL_PROXY_API_KEY` for access.
4.  Create a `.env` file in the project root and add your configuration:

    ```env
    MODEL_PROXY_URL=https://mp-staging.kaggle.net/models/openapi
    MODEL_PROXY_API_KEY={your_token}
    LLM_DEFAULT=google/gemini-2.5-flash
    LLM_DEFAULT_EVAL=google/gemini-2.5-pro
    LLMS_AVAILABLE=anthropic/claude-sonnet-4,google/gemini-2.5-flash,meta/llama-3.1-70b,google/gemini-2.5-pro
    PYTHONPATH=src
    ```

    - `LLM_DEFAULT`: Sets the model identifier for `kbench.llm`, the default model used for running tasks.
    - `LLM_DEFAULT_EVAL`: Sets the model identifier for `kbench.judge_llm`, which is typically a model used for evaluation or judging the outputs of other models.
    - `LLMS_AVAILABLE`: A comma-separated list of models authorized for use by your proxy token.

    **Note**: The `LLM_DEFAULT`, `LLM_DEFAULT_EVAL`, and `LLMS_AVAILABLE` variables depend on the models authorized by your proxy token.

**Caching**:

To speed up development and reduce costs, the framework uses [hishel](https://hishel.com/) to cache HTTP responses. Caching is disabled by default. To enable it, set the `ENABLE_LOCAL_CACHING` environment variable to `true`:

```bash
export ENABLE_LOCAL_CACHING=true
```
