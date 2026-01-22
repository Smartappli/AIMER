import sys
import platform
import torch


def _safe(label, fn, default="N/A"):
    try:
        return fn()
    except Exception as e:
        return f"{default} ({type(e).__name__}: {e})"


print(
    f"Python: {sys.version.split()[0]} ({platform.system()} {platform.release()})"
)
print(f"PyTorch version: {torch.__version__}")

# --- Build info (CUDA / HIP (ROCm) / etc.)
print(f"torch.version.cuda: {torch.version.cuda}")
print(f"torch.version.hip:  {getattr(torch.version, 'hip', None)}")
print(f"torch.version.git:  {getattr(torch.version, 'git_version', None)}")

# --- CUDA / ROCm (AMD uses torch.cuda API too)
print("\n=== CUDA / ROCm ===")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {_safe('cuda_count', torch.cuda.device_count, 0)}")
if torch.cuda.is_available():
    cur = torch.cuda.current_device()
    print(f"CUDA current device: {cur}")
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        props = torch.cuda.get_device_properties(i)
        cap = getattr(props, "major", None), getattr(props, "minor", None)
        mem_gb = getattr(props, "total_memory", 0) / (1024**3)
        print(
            f" - [{i}] {name} | capability={cap[0]}.{cap[1]} | VRAM={mem_gb:.2f} GB"
        )

# --- Apple MPS (Mac)
print("\n=== MPS (Apple Silicon) ===")
mps_backend = getattr(torch.backends, "mps", None)
if mps_backend is None:
    print("MPS backend: not present in this build")
else:
    print(f"MPS built:     {_safe('mps_built', mps_backend.is_built, False)}")
    print(
        f"MPS available: {_safe('mps_avail', mps_backend.is_available, False)}"
    )

# --- Intel XPU (oneAPI / Intel GPU)
print("\n=== XPU (Intel) ===")
xpu = getattr(torch, "xpu", None)
if xpu is None:
    print("XPU backend: not present in this build (no torch.xpu)")
else:
    print(f"XPU available: {_safe('xpu_avail', xpu.is_available, False)}")
    print(f"XPU device count: {_safe('xpu_count', xpu.device_count, 0)}")
    if _safe("xpu_avail", xpu.is_available, False) is True:
        cur = _safe("xpu_current", xpu.current_device, "N/A")
        print(f"XPU current device: {cur}")
        try:
            n = xpu.device_count()
            for i in range(n):
                name = _safe(
                    "xpu_name", lambda: xpu.get_device_name(i), "Unknown"
                )
                print(f" - [{i}] {name}")
        except Exception as e:
            print(
                f" - Unable to enumerate XPU devices ({type(e).__name__}: {e})"
            )

# --- CPU basics
print("\n=== CPU ===")
print(f"Num threads: {_safe('threads', torch.get_num_threads, 'N/A')}")
print(f"Default dtype: {torch.get_default_dtype()}")
