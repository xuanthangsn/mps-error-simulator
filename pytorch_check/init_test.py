# Optional: validates whether YOUR PyTorch/Triton stack fails fast (prints a clean
# error) or appears to hang when it tries to initialize CUDA while MPS is in the
# wedged state produced by stress.sh / demo.sh. Run this from inside a container (or
# venv) that has your actual torch build and points at the same MPS pipe directory.
#
# Usage: CUDA_MPS_PIPE_DIRECTORY=... CUDA_MPS_LOG_DIRECTORY=... python init_test.py
import time

print("STARTING", flush=True)
t0 = time.time()
import torch
print(f"IMPORT_DONE +{time.time()-t0:.2f}s", flush=True)

print("CALLING cuda.is_available()", flush=True)
avail = torch.cuda.is_available()
print(f"is_available={avail} +{time.time()-t0:.2f}s", flush=True)

print("CALLING cuda.init()", flush=True)
torch.cuda.init()
print(f"INIT_DONE +{time.time()-t0:.2f}s", flush=True)

print("ALLOCATING tensor on cuda", flush=True)
x = torch.randn(4, 4, device="cuda")
y = x @ x
torch.cuda.synchronize()
print(f"TENSOR_OK +{time.time()-t0:.2f}s sum={y.sum().item()}", flush=True)
print("DONE", flush=True)
