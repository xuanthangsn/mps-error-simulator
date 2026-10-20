// MPS client used to reproduce: "a client doing something illegal on the GPU (an
// out-of-bounds / illegal memory access) puts the shared MPS server into a stuck
// state that permanently refuses new clients, until the MPS daemon is restarted."
//
// Modes:
//   client --steady <busy_ms> <refill_us>
//       A persistent "background tenant": continuously enqueues harmless busy
//       kernels forever, simulating another real workload sharing the same GPU via
//       MPS. Prints a synced heartbeat every 20 enqueues so you can see whether it
//       keeps making real forward progress (and what error, if any, it starts
//       getting) once the fault happens. On SIGTERM: stops, drains, exits 0.
//
//   client --probe
//       The minimum needed to become a brand-new MPS client: cudaSetDevice, a tiny
//       cudaMalloc, one trivial kernel, cudaDeviceSynchronize. This is exactly the
//       step a new pod/process does on startup. Prints "PROBE_OK" and exits 0 on
//       success, or prints the CUDA error and exits nonzero on failure. This call is
//       NOT expected to hang in our testing -- it fails fast with a clean error when
//       the server is stuck (see README) -- so no internal timeout is needed here;
//       wrap it with the shell `timeout` command if you want a hard ceiling anyway.
//
//   client --illegal <trap|addr> [nosync]
//       The "offender": deliberately faults the GPU.
//         trap -> PTX `trap` instruction (illegal instruction exception, Xid 13)
//         addr -> write through a wild, unmapped device pointer (MMU page fault, Xid 31)
//       By default it calls cudaDeviceSynchronize() afterwards and prints the
//       resulting CUDA error before exiting 0 (a well-behaved client that sees its
//       own fault and exits cleanly). Pass "nosync" to exit immediately after the
//       async launch instead, without ever observing the error.
//
// Compile: nvcc -O2 -o client client.cu

#include <cstdio>
#include <cstdlib>
#include <csignal>
#include <cstring>
#include <unistd.h>
#include <cuda_runtime.h>

static volatile sig_atomic_t g_stop = 0;
static void on_sigterm(int) { g_stop = 1; }

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(2); \
    } \
} while (0)

// Harmless busy-loop kernel used by --steady to simulate real GPU work.
__global__ void busy_kernel(unsigned long long iters, int *out) {
    unsigned long long x = threadIdx.x + 1;
    for (unsigned long long i = 0; i < iters; ++i) {
        x = x * 2654435761ull + 1;
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) *out = (int)(x & 0xff);
}

static unsigned long long iters_for_ms(int ms) {
    // Rough calibration (measured ~710k iters/ms on a GTX 1650 Ti @ driver 580.178.04).
    // Exact timing doesn't matter here -- --steady just needs kernels short enough to
    // produce a heartbeat every second or so. Recalibrate on your own hardware if you
    // want --steady's heartbeat cadence to mean something specific; it isn't load-bearing
    // for the fault reproduction itself.
    return (unsigned long long)ms * 710000ull;
}

// Deliberately faults the GPU. Two distinct fault classes:
//   trap : illegal instruction exception (classically Xid 13).
//   addr : MMU page fault from a wild pointer write (classically Xid 31).
__global__ void trap_kernel() {
    asm volatile("trap;");
}
__global__ void illegal_addr_kernel() {
    int *p = (int *)0xdeadbeef00ULL;
    *p = 42;
}

int illegal_mode(const char *mode, bool sync_after) {
    CUDA_CHECK(cudaSetDevice(0));
    printf("ILLEGAL_LAUNCHING %s\n", mode);
    fflush(stdout);

    if (strcmp(mode, "trap") == 0) {
        trap_kernel<<<1, 1>>>();
    } else {
        illegal_addr_kernel<<<1, 1>>>();
    }

    if (sync_after) {
        cudaError_t err = cudaDeviceSynchronize();
        printf("ILLEGAL_SYNC_RESULT %s\n", cudaGetErrorString(err));
        fflush(stdout);
        printf("ILLEGAL_DONE\n");
        fflush(stdout);
        return 0;
    }

    printf("ILLEGAL_EXITING_UNSYNCED\n");
    fflush(stdout);
    return 0;
}

int steady_mode(int busy_ms, int refill_us) {
    signal(SIGTERM, on_sigterm);

    CUDA_CHECK(cudaSetDevice(0));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    int *d_out;
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(int)));

    unsigned long long iters = iters_for_ms(busy_ms);
    printf("READY\n");
    fflush(stdout);

    long n = 0;
    while (!g_stop) {
        busy_kernel<<<4, 128, 0, stream>>>(iters, d_out);
        // Sync periodically (not every launch) so a heartbeat genuinely proves this
        // tenant's work is completing on the GPU, not just being enqueued -- the
        // signal we need to tell "still running fine" from "silently stuck".
        if (++n % 20 == 0) {
            cudaError_t e = cudaStreamSynchronize(stream);
            printf("STEADY_HEARTBEAT %ld %s\n", n, cudaGetErrorString(e));
            fflush(stdout);
        }
        if (refill_us > 0) usleep(refill_us);
    }

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cudaFree(d_out);
    printf("GRACEFUL_EXIT\n");
    fflush(stdout);
    return 0;
}

int probe_mode() {
    CUDA_CHECK(cudaSetDevice(0));
    int *d_out;
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(int)));
    busy_kernel<<<1, 32>>>(1000, d_out);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(d_out);
    printf("PROBE_OK\n");
    fflush(stdout);
    return 0;
}

int main(int argc, char **argv) {
    if (argc >= 2 && strcmp(argv[1], "--steady") == 0) {
        int busy_ms = argc > 2 ? atoi(argv[2]) : 50;
        int refill_us = argc > 3 ? atoi(argv[3]) : 5000;
        return steady_mode(busy_ms, refill_us);
    } else if (argc >= 2 && strcmp(argv[1], "--probe") == 0) {
        return probe_mode();
    } else if (argc >= 3 && strcmp(argv[1], "--illegal") == 0) {
        const char *mode = argv[2];
        bool sync_after = !(argc > 3 && strcmp(argv[3], "nosync") == 0);
        return illegal_mode(mode, sync_after);
    } else {
        fprintf(stderr, "usage: %s --steady <busy_ms> <refill_us> | --probe | --illegal <trap|addr> [nosync]\n", argv[0]);
        return 1;
    }
}
