import os

import torch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def configure_cpu_threads(requested_threads: int | None = None) -> int:
    if requested_threads is None or requested_threads <= 0:
        detected = os.cpu_count() or 4
        requested_threads = max(1, min(8, detected - 2))
    torch.set_num_threads(requested_threads)
    return requested_threads
