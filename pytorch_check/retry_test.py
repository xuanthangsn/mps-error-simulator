# Optional: demonstrates why the wedge LOOKS like an infinite hang in production even
# though the raw CUDA call fails fast. Any code that retries a "device not ready"-
# looking error (very common, reasonable practice) will retry forever here, because
# the underlying state never clears on its own -- only a daemon restart fixes it.
#
# Usage: CUDA_MPS_PIPE_DIRECTORY=... CUDA_MPS_LOG_DIRECTORY=... python retry_test.py
import time
import torch

print("STARTING retry loop (simulates a service/readiness-probe that retries on error)", flush=True)
t0 = time.time()
for attempt in range(1, 16):
    try:
        x = torch.randn(4, 4, device="cuda")
        _ = (x @ x).sum().item()
        print(f"attempt {attempt}: SUCCESS +{time.time()-t0:.2f}s", flush=True)
        break
    except Exception as e:
        print(f"attempt {attempt}: FAILED +{time.time()-t0:.2f}s err={e}", flush=True)
        time.sleep(1)
else:
    print(f"GAVE_UP_AFTER_15_ATTEMPTS +{time.time()-t0:.2f}s -- a service that doesn't give up would still be retrying right now", flush=True)
print("DONE", flush=True)
