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

FROM mcr.microsoft.com/devcontainers/python:1-3.11-bullseye

RUN curl -LO https://github.com/protocolbuffers/protobuf/releases/download/v32.1/protoc-32.1-linux-x86_64.zip && \
    unzip protoc-32.1-linux-x86_64.zip -d /usr && \
    rm protoc-32.1-linux-x86_64.zip


COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/


RUN pip install playwright && \
    playwright install chromium --only-shell --with-deps
