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

ARG PROTOBUF_VERSION="29.6"

RUN curl -LO https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOBUF_VERSION}/protoc-${PROTOBUF_VERSION}-linux-x86_64.zip && \
    unzip protoc-${PROTOBUF_VERSION}-linux-x86_64.zip -d /usr && \
    rm protoc-${PROTOBUF_VERSION}-linux-x86_64.zip

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN rm -f /etc/apt/sources.list.d/yarn.list


RUN pip install playwright && \
    playwright install chromium --only-shell --with-deps
