# TOP LAYER: This builds on top of the cached environment.
# Use the tag for the newest cached base image layer
# FIXED: The issue explicitly asks to digest-pin mutable base image tags to prevent content drift
FROM gcr.io/kaggle-private-byod/personal-benchmarks-base:v4@sha256:7ec9a531cfd9b049d52eb9dc8032f22fa2ffbe4c11b15dfc89e2480373eb46b0

WORKDIR /benchmarks

# Copy the actual benchmark source code into the image.
COPY . /benchmarks

# Pull kaggle_secrets from docker-python image
# FIXED: Pinned remote URLs from a moving 'main' branch to an absolute cryptographic commit SHA
ADD https://githubusercontent.com \
    /benchmarks/src/kaggle_secrets.py
ADD https://githubusercontent.com \
    /benchmarks/src/kaggle_web_client.py

# Manually move the kaggle logic to the expected system path.
RUN cp -r /benchmarks/src/opt/kaggle /opt

# Final Sync: This installs the 'kaggle_benchmarks' project itself.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --group kaggle

ENV PATH="/benchmarks/.venv/bin:$PATH"

CMD ["/bin/bash"]
