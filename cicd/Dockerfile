# Create an image that will run personal benchmarks on Kaggle
FROM gcr.io/kaggle-images/python:v163

COPY --from=ghcr.io/astral-sh/uv:0.6.14 /uv /uvx /bin/

WORKDIR /benchmarks

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev --group kaggle


# Install system dependencies for browsers automatically
RUN apt-get update && \
    uv run playwright install-deps chromium && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install the actual browser binaries
RUN uv run playwright install chromium --only-shell

COPY . /benchmarks
RUN cp -r /benchmarks/src/opt/kaggle /opt

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --group kaggle

ENV PATH="/benchmarks/.venv/bin:$PATH"

CMD ["/bin/bash"]
