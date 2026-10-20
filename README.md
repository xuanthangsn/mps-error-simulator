# MPS illegal-access fault reproduction

Reproduces: **a client that performs an illegal GPU memory access (an out-of-bounds
device-pointer write) permanently wedges the shared MPS server for every other
client, until the MPS daemon is restarted — but only if at least one other client is
connected to MPS at the moment of the fault.** A lone offender with nobody else
connected recovers on its own instantly; harmless.

Confirmed on a dev box (consumer GPU, driver 580.178.04): the fault produces a real
`NVRM: Xid 31` MMU fault in the kernel log, the MPS server log shows it tearing down
*every* connected client and scheduling a device reset, and from that point on it
refuses every new client with a clean, fast `error 807` — forever, never recovering
on its own. Already-connected clients don't crash either; they keep running their own
loop but every subsequent GPU call returns `an illegal memory access was encountered`.
Restarting only the MPS control daemon (no GPU reset, no reboot) fully recovers it.

**Why this looks like a "hang" in production**: the raw CUDA call fails fast and
clean, it doesn't block. But the stuck state never clears by itself, so anything that
retries on a transient-looking GPU error (readiness probes, a service's own retry
logic, Triton's model-load retry path) retries forever — indistinguishable from a
hang from the outside. See `pytorch_check/` to confirm this against your own stack.

## ⚠️ Safety — read before running on a production box

This script **deliberately crashes a GPU context and, with ≥1 other client
connected, wedges the whole MPS server for that GPU.** Treat it like you would a
chaos-engineering fault injection:

- **Do not point it at the MPS daemon serving live traffic.** Run it against a
  cordoned node, a GPU not currently taking real requests, or a second/private MPS
  daemon on the same box (`MPS_PIPE_DIR`/`MPS_LOG_DIR` below let you do this without
  touching the production one).
- The fault only breaks the **MPS server's client bookkeeping**, not the GPU/driver
  itself — in our testing the display and `nvidia-smi` stayed fully healthy throughout,
  and recovery never required a GPU reset or reboot. But datacenter driver/hardware
  behavior could differ; don't assume this.
- Every trial requires restarting the MPS daemon to clean up (the scripts do this for
  you), which drops **every** client currently attached to that daemon — including
  any unrelated real workload.

## Prerequisites

- NVIDIA driver + `nvidia-cuda-mps-control` (ships with the driver).
- CUDA toolkit (`nvcc`) to build the client.
- Docker + [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)
  registered as a runtime (`nvidia-ctk runtime configure --runtime=docker && systemctl restart docker`).
- Read access to the kernel log (`journalctl -k`, e.g. via the `adm` group, or
  `dmesg`) if you want Xid confirmation in `stress.sh` — optional, it degrades
  gracefully without it.

## Quick start

```bash
./build.sh      # compiles client.cu, builds the mps-fault-repro-client image
./demo.sh       # narrated, interactive, one-shot walkthrough of the bug + recovery
```

`demo.sh` starts its own private MPS daemon under `/tmp/nvidia-mps` by default (safe,
isolated from production). Override with:

```bash
MPS_PIPE_DIR=/var/lib/nvidia/mps MPS_LOG_DIR=/var/log/nvidia-mps MANAGE_DAEMON=0 ./demo.sh
```

`MANAGE_DAEMON=0` means the script will **not** start/stop/restart the daemon for
you — use this only against a daemon you've deliberately stood up for this test, and
restart it yourself when the script tells you to.

## Statistical run

```bash
./stress.sh 0 10   # baseline: 10 trials, 0 concurrent tenants -> expect 0 wedges
./stress.sh 1 30   # 30 trials, 1 concurrent tenant -> expect most/all to wedge
./stress.sh 6 30   # same, with 6 tenants (closer to a real multi-tenant GPU)
```

`stress.sh` always manages its **own** private daemon (see safety note above —
intentionally, so you can't accidentally point the repeated-wedge-and-restart loop at
a live one). Results append to `results.csv`; columns include whether a real Xid was
logged, whether the "innocent" tenants kept progressing, and a verdict:

| verdict | meaning |
|---|---|
| `HEALTHY` | recovered cleanly, no bug |
| `REFUSED_807` | the bug: permanently refused with a clean, fast error |
| `HANG_AT_CLIENT_ONLY` | the bug, but manifesting as an actual block on the new-client call |
| `WEDGE` | the bug, and even the control daemon's own command socket stopped responding |

All three non-`HEALTHY` verdicts are "the bug" — they differ only in which layer
shows the symptom, not in severity or root cause.

## Confirming it against your real PyTorch/Triton stack

Once `stress.sh`/`demo.sh` has put MPS into the stuck state (and *before* you restart
the daemon), run your actual framework against the same `CUDA_MPS_PIPE_DIRECTORY`:

```bash
CUDA_MPS_PIPE_DIRECTORY=$MPS_PIPE_DIR CUDA_MPS_LOG_DIRECTORY=$MPS_LOG_DIR \
  python pytorch_check/init_test.py     # does it fail fast, or actually block?

CUDA_MPS_PIPE_DIRECTORY=$MPS_PIPE_DIR CUDA_MPS_LOG_DIRECTORY=$MPS_LOG_DIR \
  python pytorch_check/retry_test.py    # proves a retry loop never gives up on its own
```

In our testing, PyTorch 2.5.1 (single- and multi-threaded) always failed fast with a
clean `RuntimeError`, same as the raw CUDA probe — no framework-level deadlock. If
your production "hang" is real (process genuinely frozen, not just repeatedly
failing), that points to retry/readiness-probe logic somewhere in your stack rather
than PyTorch itself — `retry_test.py` demonstrates exactly how that happens.

## Recovery

```bash
echo quit | CUDA_MPS_PIPE_DIRECTORY=$MPS_PIPE_DIR nvidia-cuda-mps-control
pkill -9 -f nvidia-cuda-mps-control
pkill -9 -f nvidia-cuda-mps-server
CUDA_MPS_PIPE_DIRECTORY=$MPS_PIPE_DIR CUDA_MPS_LOG_DIRECTORY=$MPS_LOG_DIR nvidia-cuda-mps-control -d
```

No GPU reset or reboot needed in our testing — restarting the daemon (which also
respawns the server) was sufficient every time.

## Files

- `client.cu` / `Dockerfile` / `build.sh` — the test client and its container image.
  Three modes: `--steady` (a harmless background tenant), `--probe` (a minimal new
  client), `--illegal trap|addr` (the offender).
- `demo.sh` — one-shot narrated walkthrough.
- `stress.sh` — statistical run over many trials.
- `pytorch_check/` — optional validation against a real ML framework.

## Scope note

This repo tests one specific hypothesis: an illegal-access GPU fault with concurrent
clients wedging MPS. A separate, earlier hypothesis — that merely SIGKILLing a client
mid-flight (no illegal access involved) wedges MPS via a daemon bookkeeping race — was
tested extensively (serialized and concurrent, ~385 trials) and did **not**
reproduce on this hardware. That harness isn't included here since it came back
negative; ask if you want it too.
