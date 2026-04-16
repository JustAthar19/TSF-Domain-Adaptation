import os
import torch

def setup_runtime():
    cpu_cores = os.cpu_count() or 4
    default_threads = min(24, cpu_cores)

    os.environ.setdefault("OMP_NUM_THREADS", str(default_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(default_threads))

    torch.set_num_threads(default_threads)
    torch.set_num_interop_threads(max(1, min(4, default_threads // 2)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    return device
