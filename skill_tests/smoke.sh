#!/usr/bin/env bash
# Smoke test for the skill-test harness. Runs three short scenarios so CI can
# verify the harness still works end-to-end without paying for the full 54-scenario sweep.
#
# Required env: ANTHROPIC_API_KEY
# Usage: bash skill_tests/smoke.sh [extra args forwarded to run_skill_tests.py]
set -euo pipefail

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "ANTHROPIC_API_KEY is not set; cannot run smoke test." >&2
  exit 2
fi

cd "$(dirname "$0")/.."

SCENARIOS="${SMOKE_SCENARIOS:-1.1,1.9,6.1}"
OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-skill_tests/results/smoke-$(date -u +%Y%m%dT%H%M%SZ)}"

echo ">>> Running smoke scenarios: ${SCENARIOS}"
echo ">>> Writing to: ${OUTPUT_DIR}"

python3 skill_tests/run_skill_tests.py \
  --scenarios "${SCENARIOS}" \
  --output-dir "${OUTPUT_DIR}" \
  --parallel 3 \
  "$@"

echo ">>> Smoke test complete. Report:"
echo "    ${OUTPUT_DIR}/report.md"
