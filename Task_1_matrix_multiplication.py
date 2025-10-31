import time
import os
import gc
import shutil
import psutil
import numpy as np
import ray
import torch

# ---------- CLEAN OLD RAY SESSIONS ----------
def cleanup_ray_tmp():
    tmp_dir = "/tmp/ray"
    if os.path.exists(tmp_dir):
        for item in os.listdir(tmp_dir):
            item_path = os.path.join(tmp_dir, item)
            if os.path.isdir(item_path) and item.startswith("session_"):
                try:
                    shutil.rmtree(item_path)
                    print(f"Cleaned old Ray session: {item}")
                except Exception as e:
                    print(f"Could not clean {item}: {e}")

cleanup_ray_tmp()
# -------------------------------------------

# ---------- CONFIG ----------
N                 = 2048
NUM_CPU_TASKS     = 4
NUM_MPS_TASKS     = 1
OBJECT_STORE_MB   = 200
# ---------------------------

def human(x: float) -> str:
    return f"{x:.3f}"

def mem_snapshot():
    p = psutil.Process(os.getpid())
    return psutil.cpu_percent(interval=None), p.memory_info().rss / (1024**2)

def mps_available():
    return torch.backends.mps.is_available() and torch.backends.mps.is_built()

# ----------------------------------------------------------------------
if __name__ == "__main__":
    print(f"PyTorch MPS: {mps_available()}")

    ray.init(
        num_cpus=psutil.cpu_count(logical=False),
        object_store_memory=OBJECT_STORE_MB * 1024**2,
        resources={"MPS": 1.0},
        ignore_reinit_error=True,
    )

    A = np.random.randn(N, N).astype(np.float32)
    B = np.random.randn(N, N).astype(np.float32)
    print(f"N = {N} → {A.nbytes/1e6:.1f} MB per matrix")

    a_ref = ray.put(A)
    b_ref = ray.put(B)

    @ray.remote
    def cpu_matmul(a, b):
        t0 = time.time()
        c = a.dot(b)
        return {"time": time.time() - t0, "sum": float(np.sum(c))}

    @ray.remote(resources={"MPS": 1.0})
    def mps_matmul(a, b):
        device = torch.device("mps")
        a = np.copy(a)  # Force writable
        b = np.copy(b)
        a_t = torch.from_numpy(a).to(device)
        b_t = torch.from_numpy(b).to(device)

        torch.mps.synchronize()
        t0 = time.time()
        c_t = torch.matmul(a_t, b_t)
        c_cpu = c_t.to("cpu")
        t1 = time.time()
        return {"time": t1 - t0, "sum": float(torch.sum(c_cpu).item())}

    # --- Warm-up ---
    print("\n--- Warm-up ---")
    _ = ray.get(cpu_matmul.remote(a_ref, b_ref))
    if mps_available():
        torch.mps.empty_cache()
        _ = ray.get(mps_matmul.remote(a_ref, b_ref))

    # --- CPU-only ---
    print(f"\n--- CPU-only ({NUM_CPU_TASKS} tasks) ---")
    torch.mps.empty_cache()
    st = time.time()
    cpu_futs = [cpu_matmul.remote(a_ref, b_ref) for _ in range(NUM_CPU_TASKS)]
    cpu_res = ray.get(cpu_futs)
    cpu_wall = time.time() - st
    print(f"Wall: {human(cpu_wall)} s | Per-task: {human(sum(r['time'] for r in cpu_res)/len(cpu_res))}")

    # --- MPS-only ---
    print(f"\n--- MPS-only ({NUM_MPS_TASKS} task) ---")
    torch.mps.empty_cache()
    st = time.time()
    mps_futs = [mps_matmul.remote(a_ref, b_ref) for _ in range(NUM_MPS_TASKS)]
    mps_res = ray.get(mps_futs)
    mps_wall = time.time() - st
    print(f"Wall: {human(mps_wall)} s | Time: {human(mps_res[0]['time'])} s")

    # --- Heterogeneous ---
    print(f"\n--- Heterogeneous (CPU {NUM_CPU_TASKS} + MPS {NUM_MPS_TASKS}) ---")
    torch.mps.empty_cache()
    st = time.time()
    futures = (
        [cpu_matmul.remote(a_ref, b_ref) for _ in range(NUM_CPU_TASKS)] +
        [mps_matmul.remote(a_ref, b_ref) for _ in range(NUM_MPS_TASKS)]
    )
    results = ray.get(futures)
    hetero_wall = time.time() - st

    cpu_times = [r["time"] for r in results[:NUM_CPU_TASKS]]
    mps_times = [r["time"] for r in results[NUM_CPU_TASKS:]]
    print(f"Wall: {human(hetero_wall)} s")
    for i, t in enumerate(cpu_times): print(f"  CPU {i}: {human(t)} s")
    for i, t in enumerate(mps_times): print(f"  MPS {i}: {human(t)} s")

    # --- Summary ---
    print("\n" + "="*55)
    print(f"{'Mode':<15} {'Wall (s)':>10} {'Speedup vs CPU':>20}")
    print("-"*55)
    print(f"{'CPU-only':<15} {human(cpu_wall):>10} {'—':>20}")
    print(f"{'MPS-only':<15} {human(mps_wall):>10} {human(cpu_wall/mps_wall):>20}x")
    print(f"{'Heterogeneous':<15} {human(hetero_wall):>10} {human(cpu_wall/hetero_wall):>20}x")
    print("="*55)

    # --- Plot ---
    try:
        import matplotlib.pyplot as plt
        plt.bar(['CPU-only', 'MPS-only', 'Hetero'], [cpu_wall, mps_wall, hetero_wall])
        plt.ylabel('Wall Time (s)')
        plt.title('M1 Air 8 GB – Heterogeneous MatMul')
        plt.savefig('results/Task1/m1_Matrix_multiplication_benchmark.png', dpi=150, bbox_inches='tight')
        plt.show()
    except: pass

    # --- Cleanup ---
    print(f"Final: CPU {mem_snapshot()[0]}%, RSS {mem_snapshot()[1]:.1f} MB")
    torch.mps.empty_cache()
    gc.collect()
    ray.shutdown()
    print("Done.")