#!/usr/bin/env bash
# One-shot, narrated demonstration of the bug. Not a statistics-gathering tool (see
# stress.sh for that) -- this is the "watch it happen" script: start a healthy MPS
# setup, prove a lone fault recovers cleanly, then prove the SAME fault with one
# extra concurrent client wedges the server permanently until you restart it.
#
# Safety: this launches a container that deliberately faults the GPU. See README.md
# before running this anywhere near real traffic.
#
# Env vars (override to point at your real MPS directories):
#   MPS_PIPE_DIR (default /tmp/nvidia-mps)
#   MPS_LOG_DIR  (default /tmp/nvidia-log)
#   MANAGE_DAEMON=1 (default) -- this script starts/stops its OWN private MPS daemon
#                     using the directories above. Set to 0 if you want to run this
#                     against an MPS daemon that's already running (e.g. the
#                     production one) -- in that case the script will NOT start,
#                     stop, or restart it for you; you do that yourself, deliberately,
#                     at the point the script tells you to.

set -u
cd "$(dirname "$0")"

MPS_PIPE_DIR="${MPS_PIPE_DIR:-/tmp/nvidia-mps}"
MPS_LOG_DIR="${MPS_LOG_DIR:-/tmp/nvidia-log}"
MANAGE_DAEMON="${MANAGE_DAEMON:-1}"
IMAGE=mps-fault-repro-client

log()  { echo; echo "=== $* ==="; }
note() { echo "    $*"; }

run_client() {
    docker run --rm --gpus all --user "$(id -u):$(id -g)" \
        -e CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps \
        -e CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log \
        -v "$MPS_PIPE_DIR":/tmp/nvidia-mps \
        -v "$MPS_LOG_DIR":/tmp/nvidia-log \
        "$@"
}

run_client_detached() {
    local name="$1"; shift
    docker run -d --name "$name" --gpus all --user "$(id -u):$(id -g)" \
        -e CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps \
        -e CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log \
        -v "$MPS_PIPE_DIR":/tmp/nvidia-mps \
        -v "$MPS_LOG_DIR":/tmp/nvidia-log \
        "$@" >/dev/null
}

start_daemon() {
    mkdir -p "$MPS_PIPE_DIR" "$MPS_LOG_DIR"
    CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE_DIR" CUDA_MPS_LOG_DIRECTORY="$MPS_LOG_DIR" nvidia-cuda-mps-control -d
    sleep 1
}

stop_daemon() {
    CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE_DIR" bash -c 'echo quit | timeout 5 nvidia-cuda-mps-control' >/dev/null 2>&1
    sleep 1
    pkill -9 -f nvidia-cuda-mps-control 2>/dev/null
    pkill -9 -f nvidia-cuda-mps-server 2>/dev/null
    sleep 1
}

daemon_alive() {
    CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE_DIR" bash -c 'echo get_server_list | timeout 5 nvidia-cuda-mps-control' >/dev/null 2>&1
}

cleanup() {
    docker rm -f demo-steady demo-offender >/dev/null 2>&1
}
trap cleanup EXIT

docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "Run ./build.sh first."; exit 1; }

if [[ "$MANAGE_DAEMON" == "1" ]]; then
    log "Starting a private MPS daemon (pipe=$MPS_PIPE_DIR, log=$MPS_LOG_DIR)"
    stop_daemon
    start_daemon
    daemon_alive && note "daemon is up and responding" || { echo "daemon failed to start"; exit 1; }
else
    log "Using the MPS daemon already running at pipe=$MPS_PIPE_DIR"
    daemon_alive || { echo "No MPS daemon answering at $MPS_PIPE_DIR -- check MPS_PIPE_DIR or start one."; exit 1; }
fi

log "Step 1: lone fault, no other clients connected"
note "Launching an offender that performs an illegal device-pointer write..."
run_client "$IMAGE" --illegal addr
note "Probing with a brand-new client right away..."
if run_client "$IMAGE" --probe; then
    note "RESULT: recovered immediately. A lone fault with nobody else connected is harmless here."
else
    note "RESULT: did NOT recover immediately. (Unexpected for this step -- see README.)"
fi

read -rp $'\nPress Enter to continue to Step 2 (the one that wedges it)...\n'

log "Step 2: same fault, but with one other client connected at the same time"
note "Starting one background tenant doing completely unrelated, legitimate GPU work..."
docker rm -f demo-steady >/dev/null 2>&1
run_client_detached demo-steady "$IMAGE" --steady 50 5000
sleep 2
note "Launching the SAME offender again..."
run_client "$IMAGE" --illegal addr
note "Probing with a brand-new client right away..."
if run_client "$IMAGE" --probe; then
    note "RESULT: recovered. (On your hardware/driver this specific run may not have wedged -- try again, or see stress.sh for a statistical run.)"
else
    note "RESULT: WEDGED. New client was refused. Let's confirm this isn't just transient."
    sleep 3
    note "Probing again after a 3s wait..."
    if run_client "$IMAGE" --probe; then
        note "...recovered on its own after a few seconds. Not a permanent wedge this time."
    else
        note "...still refusing new clients. This is the bug: it will stay this way until the"
        note "MPS daemon is restarted. Note the background tenant (demo-steady) is also now"
        note "getting errors on every GPU call, even though it never did anything illegal:"
        docker logs demo-steady 2>&1 | tail -3
    fi
fi

if [[ "$MANAGE_DAEMON" == "1" ]]; then
    read -rp $'\nPress Enter to restart the MPS daemon and confirm recovery...\n'
    log "Restarting the MPS daemon (the only fix once wedged)"
    stop_daemon
    start_daemon
    note "Probing once more..."
    if run_client "$IMAGE" --probe; then
        note "RESULT: recovered. Daemon restart is confirmed as the fix."
    else
        note "RESULT: still failing after a restart -- that would be a different, worse problem. Investigate further."
    fi
else
    echo
    note "MANAGE_DAEMON=0: restart your production MPS daemon yourself to recover, then re-run"
    note "this script's probe step to confirm."
fi
