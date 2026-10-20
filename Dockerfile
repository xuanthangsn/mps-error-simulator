# Minimal runtime image for the client/prober binary. Each container is a real,
# separate process reached through the MPS pipe directory bind-mounted from the
# host -- the same architecture as hostPath-mounting /var/lib/nvidia/mps into pods.
ARG CUDA_RUNTIME_TAG=12.6.3-runtime-ubuntu24.04
FROM nvidia/cuda:${CUDA_RUNTIME_TAG}

COPY client /usr/local/bin/client
RUN chmod +x /usr/local/bin/client

ENTRYPOINT ["/usr/local/bin/client"]
