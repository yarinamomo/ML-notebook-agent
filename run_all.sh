#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON="python"

API_KEYS="api_keys_gpt.txt" # api_keys_tu
MODEL="gpt-5.3-codex"

CONFIGS=(
  "./config/agent.yaml"
  "./config/agent_without_run_code_and_output.yaml"
  "./config/baseline_without_cell_outputs.yaml"
)

log_timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

file_timestamp() {
  date '+%Y-%m-%d_%H-%M-%S'
}

human_time() {
  local seconds="$1"
  printf "%02d:%02d:%02d" \
    $((seconds / 3600)) \
    $(((seconds % 3600) / 60)) \
    $((seconds % 60))
}

MASTER_LOG="run_all_$(file_timestamp).log"

# FD 3 is only for the bash wrapper log.
exec 3>>"$MASTER_LOG"

log() {
  printf '[%s] %s\n' "$(log_timestamp)" "$*" >&3
}

total=${#CONFIGS[@]}
completed=0
failed=0
global_start=$(date +%s)

log "Starting runs"
log "API keys: $API_KEYS"
log "Model: $MODEL"
log "Total runs: $total"
log "Master log: $MASTER_LOG"
log "Python output goes directly to the terminal"
log "----------------------------------------"

for config in "${CONFIGS[@]}"; do
  completed=$((completed + 1))
  run_start=$(date +%s)

  cmd=(
    "$PYTHON" -u "./main.py"
    --threads
    --api-keys "$API_KEYS"
    --model "$MODEL"
    -c "$config"
  )

  log "Config $completed/$total starting"
  log "Config: $config"
  log "Command: $(printf '%q ' "${cmd[@]}")"

  # Python stdout/stderr goes directly to your current terminal/tmux pane.
  if "${cmd[@]}"; then
    status=0
  else
    status=$?
    failed=1
  fi

  run_end=$(date +%s)
  run_seconds=$((run_end - run_start))
  elapsed=$((run_end - global_start))
  remaining=$((total - completed))

  avg=$((elapsed / completed))
  eta_seconds=$((avg * remaining))

  log "Config $completed/$total finished with status $status"
  log "duration: $(human_time "$run_seconds")"
  log "Elapsed total: $(human_time "$elapsed")"
  log "Estimated remaining: $(human_time "$eta_seconds")"
  log "----------------------------------------"

  if [[ "$status" -eq 0 ]]; then
    patched_cmd=(
      "$PYTHON" -u "./data_analysis/run_all_patched_notebooks.py"
      -c "$config"
    )

    log "Try to rerun missing patched notebooks for config $completed/$total"
    log "Config: $config"
    log "Command: $(printf '%q ' "${patched_cmd[@]}")"

    # Patched notebook stdout/stderr goes directly to your current terminal/tmux pane.
    "${patched_cmd[@]}"

    log "Config $completed/$total patched run finished"
    log "----------------------------------------"
  fi
done

total_seconds=$(($(date +%s) - global_start))

log "All runs finished"
log "Total duration: $(human_time "$total_seconds")"
log "Failed flag: $failed"

exit "$failed"