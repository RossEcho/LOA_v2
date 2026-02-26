#!/usr/bin/env sh
set -u

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
LOG_DIR="$ROOT_DIR/.logs"
mkdir -p "$LOG_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/test_run_${TS}.log"

TEST_TOOL="${TEST_TOOL:-ping}"
PYTHON_BIN="${PYTHON_BIN:-python}"

PASS_COUNT=0
FAIL_COUNT=0

log() {
  printf '%s\n' "$1" | tee -a "$LOG_FILE"
}

run_step() {
  NAME="$1"
  shift
  log ""
  log "=== $NAME ==="
  log "CMD: $*"
  if "$@" >>"$LOG_FILE" 2>&1; then
    PASS_COUNT=$((PASS_COUNT + 1))
    log "RESULT: PASS"
    return 0
  fi
  FAIL_COUNT=$((FAIL_COUNT + 1))
  log "RESULT: FAIL"
  return 1
}

run_expected_fail_step() {
  NAME="$1"
  shift
  log ""
  log "=== $NAME ==="
  log "CMD (expected to fail): $*"
  if "$@" >>"$LOG_FILE" 2>&1; then
    FAIL_COUNT=$((FAIL_COUNT + 1))
    log "RESULT: FAIL (unexpected success)"
    return 1
  fi
  PASS_COUNT=$((PASS_COUNT + 1))
  log "RESULT: PASS (failed as expected)"
  return 0
}

log "Test run started: $TS"
log "Repo: $ROOT_DIR"
log "Log: $LOG_FILE"

cd "$ROOT_DIR" || exit 1

run_step "Python unittest suite" env PYTHONDONTWRITEBYTECODE=1 "$PYTHON_BIN" -m unittest
run_step "Bridge list-tools smoke" "$PYTHON_BIN" bin/loa-bridge --list-tools

run_step "Bridge ping call smoke" sh -c \
  "printf '%s' '{\"tool_name\":\"ping\",\"args\":{\"target\":\"8.8.8.8\",\"count\":1},\"cwd\":null,\"timeout_seconds\":10,\"action_class\":\"NETWORK\",\"env\":null}' | \"$PYTHON_BIN\" bin/loa-bridge"

run_step "Onboarding registry list" "$PYTHON_BIN" main.py --list-tools
run_expected_fail_step "Missing tool-spec returns JSON error" "$PYTHON_BIN" main.py --tool-spec does-not-exist

if command -v "$TEST_TOOL" >/dev/null 2>&1; then
  run_step "Init tool onboarding for $TEST_TOOL" "$PYTHON_BIN" main.py --init-tool "$TEST_TOOL"
  run_step "Read tool spec for $TEST_TOOL" "$PYTHON_BIN" main.py --tool-spec "$TEST_TOOL"
else
  log ""
  log "=== Onboarding smoke skipped ==="
  log "Tool '$TEST_TOOL' not found in PATH."
fi

log ""
log "=== Summary ==="
log "Passed: $PASS_COUNT"
log "Failed: $FAIL_COUNT"
log "Log file: $LOG_FILE"

if [ "$FAIL_COUNT" -gt 0 ]; then
  exit 1
fi
exit 0