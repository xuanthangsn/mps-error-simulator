#!/usr/bin/env bash
# Statistical version of demo.sh: runs many trials of "illegal access + N concurrent
# tenants" and reports how often it wedges the MPS server, how often an old/innocent
# tenant starts erroring, and whether a real Xid fault was logged each time.
#
# Unlike demo.sh, this always starts and owns its OWN private MPS daemon (under
# MPS_PIPE_DIR/MPS_LOG_DIR) rather than attaching to an existing one -- on a shared
# production box, point this at an otherwise-idle GPU or a cordoned node, not at the
# daemon serving live traffic. See README.md.
#
# Usage: ./stress.sh [n_steady_tenants] [trials]
#   n_steady_tenants: concurrent innocent clients running during each trial (try 0
#                      first to confirm the "lone fault is harmless" baseline, then
#                      1+ to reproduce the wedge).
#   trials: number of illegal-access trials to run.

set -u
cd "$(dirname "$0")"

MPS_PIPE_DIR="${MPS_PIPE_DIR:-/tmp/nvidia-mps}"
MPS_LOG_DIR="${MPS_LOG_DIR:-/tmp/nvidia-log}"
IMAGE=mps-fault-repro-client
RESULTS=results.csv

N_STEADY="${1:-1}"
TRIALS="${2:-30}"
PROBE_TIMEOUT=10

log() { echo "[$(date '+%H:%M:%S')] $*" >&2; }

# --- Xid log watcher: journalctl -k if readable, else dmesg -w, else skip gracefully ---
KERNLOG="$(mktemp /tmp/mps-fault-repro-kernlog.XXXXXX)"
WATCHER_PID=""
start_kernel_watcher() {
    if journalctl -k -n0 --no-pager >/dev/null 2>&1; then
        journalctl -k -f --no-pager > "$KERNLOG" 2>&1 &
        WATCHER_PID=$!
    elif dmesg -w > "$KERNLOG" 2>&1 & then
        WATCHER_PID=$!
        sleep 0.3
        grep -q "Operation not permitted" "$KERNLOG" && { kill "$WATCHER_PID" 2>/dev/null; WATCHER_PID=""; }
    fi
    if [[ -z "$WATCHER_PID" ]]; then
        log "WARNING: can't read kernel log (need journalctl -k access, e.g. being in the 'adm' group, or dmesg). Xid confirmation will be skipped."
    fi
}
stop_kernel_watcher() { [[ -n "$WATCHER_PID" ]] && kill "$WATCHER_PID" 2>/dev/null; }

start_daemon() {
    mkdir -p "$MPS_PIPE_DIR" "$MPS_LOG_DIR"
    CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE_DIR" CUDA_MPS_LOG_DIRECTORY="$MPS_LOG_DIR" nvidia-cuda-mps-control -d
    sleep 1
    log "MPS daemon started"
}
stop_daemon() {
    CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE_DIR" bash -c 'echo quit | timeout 5 nvidia-cuda-mps-control' >/dev/null 2>&1
    sleep 1
    pkill -9 -f nvidia-cuda-mps-control 2>/dev/null
    pkill -9 -f nvidia-cuda-mps-server 2>/dev/null
    sleep 1
}
restart_daemon() { log ">>> RESTARTING MPS daemon"; stop_daemon; start_daemon; }

docker_run_client() {
    local name="$1"; shift
    docker run -d --name "$name" --gpus all --user "$(id -u):$(id -g)" \
        -e CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps -e CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log \
        -v "$MPS_PIPE_DIR":/tmp/nvidia-mps -v "$MPS_LOG_DIR":/tmp/nvidia-log \
        "$IMAGE" "$@" >/dev/null
}

start_steady_tenants() {
    for i in $(seq 1 "$N_STEADY"); do
        docker rm -f "fr-steady-$i" >/dev/null 2>&1
        docker_run_client "fr-steady-$i" --steady 50 5000
    done
    (( N_STEADY > 0 )) && sleep 2
}
stop_steady_tenants() {
    for i in $(seq 1 "$N_STEADY"); do
        docker kill -s SIGTERM "fr-steady-$i" >/dev/null 2>&1
    done
    sleep 1
    for i in $(seq 1 "$N_STEADY"); do docker rm -f "fr-steady-$i" >/dev/null 2>&1; done
}
steady_heartbeat_total() {
    local total=0
    for i in $(seq 1 "$N_STEADY"); do
        local c
        c=$(docker logs "fr-steady-$i" 2>/dev/null | grep -c '^STEADY_HEARTBEAT')
        total=$((total+c))
    done
    echo "$total"
}

probe_daemon_pipe() {
    local start=$(date +%s.%N) rc
    CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE_DIR" timeout "$PROBE_TIMEOUT" bash -c 'echo get_server_list | nvidia-cuda-mps-control' >/tmp/mfr_daemon_probe.out 2>&1
    rc=$?
    local dur=$(echo "$(date +%s.%N) - $start" | bc)
    if (( rc == 0 )); then echo "OK $dur"
    elif (( rc == 124 )); then echo "HANG $PROBE_TIMEOUT"
    else echo "FAST_ERROR_rc${rc} $dur"
    fi
}

probe_new_client() {
    local start=$(date +%s.%N) rc
    timeout "$PROBE_TIMEOUT" docker run --rm --gpus all --user "$(id -u):$(id -g)" \
        -e CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps -e CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log \
        -v "$MPS_PIPE_DIR":/tmp/nvidia-mps -v "$MPS_LOG_DIR":/tmp/nvidia-log \
        "$IMAGE" --probe >/tmp/mfr_client_probe.out 2>&1
    rc=$?
    local dur=$(echo "$(date +%s.%N) - $start" | bc)
    if (( rc == 0 )); then echo "OK $dur"
    elif (( rc == 124 )); then echo "HANG $PROBE_TIMEOUT"
    else echo "FAST_ERROR_rc${rc} $dur"
    fi
}

run_trial() {
    local idx="$1"
    local cname="fr-offender-${idx}"
    docker rm -f "$cname" >/dev/null 2>&1

    local kernlog_before hb_before
    kernlog_before=$(wc -l < "$KERNLOG" 2>/dev/null || echo 0)
    hb_before=$(steady_heartbeat_total)

    docker_run_client "$cname" --illegal addr
    local w=0
    while (( w < 150 )); do
        docker inspect -f '{{.State.Running}}' "$cname" 2>/dev/null | grep -q true || break
        sleep 0.1; w=$((w+1))
    done
    docker rm -f "$cname" >/dev/null 2>&1

    # First probe, then up to 2 confirmation retries 2s apart -- a single failed probe
    # could just be the normal brief disruption around any client disconnect; only
    # calling it the bug once it's SUSTAINED avoids false positives.
    local client_probe daemon_probe attempt
    for attempt in 1 2 3; do
        client_probe=$(probe_new_client)
        daemon_probe=$(probe_daemon_pipe)
        if ! echo "$client_probe" | grep -qE "^HANG |^FAST_ERROR" && ! echo "$daemon_probe" | grep -q "^HANG "; then
            break   # recovered -- no need to keep confirming
        fi
        (( attempt < 3 )) && sleep 2
    done

    sleep 0.5
    local hb_after steady_ok="n/a"
    hb_after=$(steady_heartbeat_total)
    if (( N_STEADY > 0 )); then
        steady_ok="yes"
        (( hb_after > hb_before )) || steady_ok="no_progress"
    fi

    local xid_seen="skipped"
    if [[ -s "$KERNLOG" ]]; then
        xid_seen="no"
        tail -n +"$((kernlog_before+1))" "$KERNLOG" | grep -q "NVRM: Xid" && xid_seen="yes"
    fi

    local verdict="HEALTHY"
    echo "$daemon_probe" | grep -q "^HANG " && verdict="WEDGE"
    [[ "$verdict" == "HEALTHY" ]] && echo "$client_probe" | grep -q "^HANG " && verdict="HANG_AT_CLIENT_ONLY"
    [[ "$verdict" == "HEALTHY" ]] && echo "$client_probe" | grep -q "^FAST_ERROR" && verdict="REFUSED_807"

    echo "$idx,xid_seen=$xid_seen,steady_progress=$steady_ok,client_probe=[$client_probe],daemon_probe=[$daemon_probe],verdict=$verdict"
}

cleanup() {
    stop_kernel_watcher
    stop_steady_tenants
    docker ps -aq --filter "name=fr-offender-" | xargs -r docker rm -f >/dev/null 2>&1
    rm -f "$KERNLOG"
}
trap cleanup EXIT

main() {
    command -v nvidia-cuda-mps-control >/dev/null || { echo "nvidia-cuda-mps-control not found"; exit 1; }
    docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "Run ./build.sh first."; exit 1; }

    start_kernel_watcher
    restart_daemon
    : > "$RESULTS"
    echo "trial,details" >> "$RESULTS"

    start_steady_tenants
    log "=== $TRIALS trials: illegal-address fault with $N_STEADY concurrent steady tenant(s) ==="

    local wedges=0 refused_807=0
    for i in $(seq 1 "$TRIALS"); do
        row=$(run_trial "$i")
        echo "$row" | tee -a /dev/stderr >> "$RESULTS"
        if echo "$row" | grep -q "verdict=WEDGE\|verdict=REFUSED_807\|verdict=HANG_AT_CLIENT_ONLY"; then
            echo "$row" | grep -q "verdict=WEDGE" && wedges=$((wedges+1))
            echo "$row" | grep -q "verdict=REFUSED_807\|verdict=HANG_AT_CLIENT_ONLY" && refused_807=$((refused_807+1))
            log "  !!! Trial $i did not recover cleanly -- restarting MPS daemon + steady tenants"
            restart_daemon
            start_steady_tenants
        fi
    done

    log "=== DONE: $wedges true hang(s), $refused_807 permanent-807-refusal case(s), out of $TRIALS trials ==="
    log "    ('permanent refusal' and 'true hang' are both the bug -- see README -- just two different failure textures)"
    log "    full detail in $RESULTS"
}

main "$@"
