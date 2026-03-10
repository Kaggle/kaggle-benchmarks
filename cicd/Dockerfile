# TOP LAYER: This builds on top of the cached environment.
# Use the tag for the newest cached base image layer
FROM gcr.io/kaggle-private-byod/personal-benchmarks-base:v2

WORKDIR /benchmarks

# Copy the actual benchmark source code into the image.
COPY . /benchmarks

# Pull kaggle_secrets from docker-python image
ADD https://raw.githubusercontent.com/Kaggle/docker-python/main/patches/kaggle_secrets.py \
    /benchmarks/src/kaggle_secrets.py
ADD https://raw.githubusercontent.com/Kaggle/docker-python/main/patches/kaggle_web_client.py \
    /benchmarks/src/kaggle_web_client.py

# Manually move the kaggle logic to the expected system path.
RUN cp -r /benchmarks/src/opt/kaggle /opt

# Final Sync: This installs the 'kaggle_benchmarks' project itself.
# Since all dependencies are already in the base image,
# uv only needs to link your local source code and compile its bytecode.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --group kaggle

ENV PATH="/benchmarks/.venv/bin:$PATH"

CMD ["/bin/bash"]
