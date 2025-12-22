# Create an image that will run personal benchmarks on Kaggle
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.6.14 /uv /uvx /bin/

WORKDIR /benchmarks

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=src,target=src \
    --mount=type=bind,source=README.md,target=README.md \
    UV_INDEX_URL=https://pypi.org/simple \
    uv sync \
      --no-install-project \
      --no-dev \
      --group kaggle

# Manually install Playwright's system dependencies for Debian Trixie.
# This step is necessary because the base image was recently updated to Debian
# Trixie, which changed the names and availability of several packages required
# by Playwright's browsers.
#
# This list of packages was obtained from the detailed error logs produced
# by Playwright when it failed to launch a browser.
RUN apt-get update && apt-get install -y \
    libnss3 libnss3-tools \
    libnspr4 \
    libdbus-1-3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libatspi2.0-0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libpango-1.0-0 \
    libcairo2 \
    libasound2 \
    fonts-unifont \
    libxkbcommon0 \
    libglib2.0-0 \
    libgobject-2.0-0 \
    libexpat1 \
    libx11-6 \
    libxext6 \
    libxcb1 \
    libgtk-3-0 \
    libfontconfig1 \
    --no-install-recommends

RUN uv run playwright install chromium --only-shell

COPY . /benchmarks
RUN cp -r /benchmarks/src/opt/kaggle /opt

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --group kaggle

ENV PATH="/benchmarks/.venv/bin:$PATH"

CMD ["/bin/bash"]
