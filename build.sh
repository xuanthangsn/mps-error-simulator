#!/usr/bin/env bash
# Compile the client on the host (same glibc/driver the containers will see via
# nvidia-container-toolkit) and build the runtime image around it.
set -euxo pipefail
cd "$(dirname "$0")"

nvcc -O2 -o client client.cu
docker build -t mps-fault-repro-client .
