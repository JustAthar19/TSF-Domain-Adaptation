import os

def get_dataloader_kwargs(device, shuffle=True, drop_last=False):
    num_workers = 0
    if device.type == "cuda":
        num_workers = min(8, max(2, os.cpu_count() // 3))

    kwargs = {
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": (device.type == "cuda"),
        "drop_last": drop_last,
    }

    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2

    return kwargs