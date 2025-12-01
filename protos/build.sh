#!/usr/bin/env bash

protoc --python_out=../src/kaggle_benchmarks/kaggle/ \
       --mypy_out=../src/kaggle_benchmarks/kaggle/ \
       *.proto
