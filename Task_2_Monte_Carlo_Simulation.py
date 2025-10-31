# Task_2_Monte_Carlo_Simulation_fixed.py
import time
import os
import math
import ray
import numpy as np
import torch
import psutil

# Tune these to keep runtime reasonable on a laptop
TOTAL_SAMPLES = 20_000_000   # reduce if you want faster runs
BATCH = 10_000_000            # each task does this many samples
# --------------------------------------------------

def setup_ray():
    # Start Ray locally and declare we have a custom "MPS" resource available.
    # We set num_cpus to the number of physical cores (or 4 if uncertain).
    num_cpus = psutil.cpu_count(logical=False) or 4
    # If you want, lower num_cpus to leave room for the system.
    ray.init(num_cpus=num_cpus, resources={"MPS": 1.0}, ignore_reinit_error=True)
    print(f"Ray started: num_cpus={num_cpus}, declared resources={{'MPS':1.0}}")

@ray.remote(num_cpus=1)
def cpu_pi(batch_samples: int, seed: int):
    # Pure-CPU worker using numpy RNG
    rng = np.random.RandomState(seed)
    x = rng.random(batch_samples)
    y = rng.random(batch_samples)
    inside = np.sum((x * x + y * y) <= 1.0)
    return inside

@ray.remote(resources={"MPS": 1.0})
def gpu_pi(batch_samples: int, seed: int):
    # GPU worker using torch with the MPS backend
    device = torch.device("mps")
    # Use torch's RNG on device for speed
    torch.manual_seed(seed)
    # create random numbers directly on device
    x = torch.rand(batch_samples, device=device)
    y = torch.rand(batch_samples, device=device)
    inside = torch.sum((x * x + y * y) <= 1.0)
    # move the single integer back to CPU (this is small)
    return int(inside.item())

def run_experiment(total_samples: int, batch: int):
    n_batches = total_samples // batch
    if n_batches < 1:
        raise ValueError("Increase total_samples or decrease batch size.")
    print(f"Running {n_batches} batches of {batch} samples each ({n_batches*batch} total)")

    # CPU-only
    start = time.time()
    cpu_tasks = [cpu_pi.remote(batch, i + 1000) for i in range(n_batches)]
    cpu_counts = ray.get(cpu_tasks)
    pi_cpu = 4.0 * sum(cpu_counts) / (n_batches * batch)
    cpu_time = time.time() - start

    # GPU-only
    start = time.time()
    gpu_tasks = [gpu_pi.remote(batch, i + 2000) for i in range(n_batches)]
    gpu_counts = ray.get(gpu_tasks)
    pi_gpu = 4.0 * sum(gpu_counts) / (n_batches * batch)
    gpu_time = time.time() - start

    # Mixed (alternate CPU / GPU tasks)
    start = time.time()
    mixed_tasks = []
    for i in range(n_batches):
        if i % 3 == 0:
            mixed_tasks.append(gpu_pi.remote(batch, i + 3000))
        else:
            mixed_tasks.append(cpu_pi.remote(batch, i + 4000))
    mixed_counts = ray.get(mixed_tasks)
    pi_mix = 4.0 * sum(mixed_counts) / (n_batches * batch)
    mixed_time = time.time() - start

    print("\nResults:")
    print(f" CPU-only  : π ≈ {pi_cpu:.6f}  time = {cpu_time:.3f} s")
    print(f" GPU-only  : π ≈ {pi_gpu:.6f}  time = {gpu_time:.3f} s")
    print(f" Hetero    : π ≈ {pi_mix:.6f}  time = {mixed_time:.3f} s")
    return {"cpu_time": cpu_time, "gpu_time": gpu_time, "mixed_time": mixed_time}

if __name__ == "__main__":
    # small safety: ensure MPS is available
    if not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
        print("Warning: PyTorch MPS backend not available. The GPU runs will fail.")
    setup_ray()
    stats = run_experiment(TOTAL_SAMPLES, BATCH)
    # cleanup
    ray.shutdown()
    print("Done. Stats:", stats)
