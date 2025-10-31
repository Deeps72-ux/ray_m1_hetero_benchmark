import time
import os
import gc
import shutil
import psutil
import numpy as np
import ray
import torch
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. Clean old Ray sessions (prevents /tmp/ray warnings)
# ----------------------------------------------------------------------
def cleanup_ray_tmp():
    tmp_dir = "/tmp/ray"
    if os.path.exists(tmp_dir):
        for item in os.listdir(tmp_dir):
            p = os.path.join(tmp_dir, item)
            if os.path.isdir(p) and item.startswith("session_"):
                try:
                    shutil.rmtree(p)
                    print(f"Cleaned old Ray session: {item}")
                except Exception:
                    pass
cleanup_ray_tmp()

# ----------------------------------------------------------------------
# 2. Configuration – safe for 8 GB M1 Air
# ----------------------------------------------------------------------
TOTAL_SAMPLES = 30_000_000          # total points
BATCH         =  5_000_000          # points per Ray task
#   → 6 tasks × 5 M = 30 M points
#   → each task uses ~40 MB (float32) → total < 300 MB in object store
OBJECT_STORE_MB = 400

# ----------------------------------------------------------------------
# 3. Ray initialisation
# ----------------------------------------------------------------------
def setup_ray():
    num_cpus = psutil.cpu_count(logical=False) or 4
    ray.init(
        num_cpus=num_cpus,
        object_store_memory=OBJECT_STORE_MB * 1024**2,
        resources={"MPS": 1.0},
        ignore_reinit_error=True,
    )
    print(f"Ray started – {num_cpus} physical cores, MPS resource declared")

# ----------------------------------------------------------------------
# 4. Workers
# ----------------------------------------------------------------------
@ray.remote(num_cpus=1)
def cpu_pi(batch_samples: int, seed: int) -> int:
    """Pure-CPU Monte-Carlo using NumPy."""
    rng = np.random.RandomState(seed)
    x = rng.random(batch_samples)
    y = rng.random(batch_samples)
    return int(np.sum(x*x + y*y <= 1.0))

@ray.remote(resources={"MPS": 1.0})
def mps_pi(batch_samples: int, seed: int) -> int:
    """GPU Monte-Carlo using torch+MPS."""
    device = torch.device("mps")
    torch.manual_seed(seed)
    # generate directly on the device – no copy overhead
    x = torch.rand(batch_samples, device=device)
    y = torch.rand(batch_samples, device=device)
    inside = torch.sum((x*x + y*y) <= 1.0)
    return int(inside.item())

# ----------------------------------------------------------------------
# 5. Experiment driver
# ----------------------------------------------------------------------
def run_experiment(total_samples: int, batch: int):
    n_batches = total_samples // batch
    if n_batches < 1:
        raise ValueError("Increase total_samples or decrease batch size.")

    print(f"\nRunning {n_batches} batches of {batch:,} samples each ({total_samples:,} total)")

    # -------------------------------------------------- CPU-only
    print("\n--- CPU-only ------------------------------------------------")
    torch.mps.empty_cache()
    st = time.time()
    cpu_futs = [cpu_pi.remote(batch, i + 1000) for i in range(n_batches)]
    cpu_counts = ray.get(cpu_futs)
    cpu_time = time.time() - st
    pi_cpu = 4.0 * sum(cpu_counts) / total_samples

    # -------------------------------------------------- MPS-only
    print("\n--- MPS-only ------------------------------------------------")
    torch.mps.empty_cache()
    st = time.time()
    mps_futs = [mps_pi.remote(batch, i + 2000) for i in range(n_batches)]
    mps_counts = ray.get(mps_futs)
    mps_time = time.time() - st
    pi_mps = 4.0 * sum(mps_counts) / total_samples

    # -------------------------------------------------- Heterogeneous
    print("\n--- Heterogeneous (CPU + MPS) -------------------------------")
    torch.mps.empty_cache()
    st = time.time()
    hetero_futs = []
    for i in range(n_batches):
        if i % 2 == 0:                     # even → MPS
            hetero_futs.append(mps_pi.remote(batch, i + 3000))
        else:                              # odd  → CPU
            hetero_futs.append(cpu_pi.remote(batch, i + 4000))
    hetero_counts = ray.get(hetero_futs)
    hetero_time = time.time() - st
    pi_hetero = 4.0 * sum(hetero_counts) / total_samples

    # -------------------------------------------------- Results
    print("\n" + "="*70)
    print(f"{'Mode':<12} {'π estimate':>12} {'Wall (s)':>10} {'Speedup':>12}")
    print("-"*70)
    print(f"{'CPU-only':<12} {pi_cpu:>12.6f} {cpu_time:>10.3f} {'—':>12}")
    print(f"{'MPS-only':<12} {pi_mps:>12.6f} {mps_time:>10.3f} {cpu_time/mps_time:>11.2f}x")
    print(f"{'Hetero':<12} {pi_hetero:>12.6f} {hetero_time:>10.3f} {cpu_time/hetero_time:>11.2f}x")
    print("="*70)

    return {
        "cpu_time": cpu_time,
        "mps_time": mps_time,
        "hetero_time": hetero_time,
        "pi_cpu": pi_cpu,
        "pi_mps": pi_mps,
        "pi_hetero": pi_hetero,
    }

# ----------------------------------------------------------------------
# 6. Plot (optional)
# ----------------------------------------------------------------------
def plot_results(stats):
    modes = ['CPU-only', 'MPS-only', 'Hetero']
    times = [stats["cpu_time"], stats["mps_time"], stats["hetero_time"]]
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    plt.figure(figsize=(7,4.5))
    bars = plt.bar(modes, times, color=colors)
    plt.ylabel('Wall Time (s)')
    plt.title('M1 Air 8 GB – Monte-Carlo π (30 M points)')
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, h + 0.02,
                 f'{h:.2f}s', ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig('results/Task2/m1_pi_benchmark.png', dpi=150)
    plt.show()

# ----------------------------------------------------------------------
# 7. Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    if not torch.backends.mps.is_available():
        print("MPS not available – GPU runs will be skipped.")
    else:
        print("MPS backend ready")

    setup_ray()
    stats = run_experiment(TOTAL_SAMPLES, BATCH)

    # optional plot
    try:
        plot_results(stats)
    except Exception as e:
        print("Plot skipped:", e)

    # cleanup
    ray.shutdown()
    print("\nRay shut down. Done.")