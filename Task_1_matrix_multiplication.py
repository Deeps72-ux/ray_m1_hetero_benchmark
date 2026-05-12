import time
import os
import gc
import shutil
import psutil
import numpy as np
import ray
import torch

# ----------------------------------------------------------------------
# 1. Clean old Ray sessions (prevents /tmp/ray warnings)
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# 2. Configuration – safe for 8 GB M1 Air
# ----------------------------------------------------------------------
N               = 4096          # Matrix dimension (N×N, float32)
NUM_BLOCKS      = 16            # Split A into this many row-blocks
OBJECT_STORE_MB = 500           # Ray object store budget

# ----------------------------------------------------------------------
# 3. Helpers
# ----------------------------------------------------------------------
def human(x: float) -> str:
    return f"{x:.3f}"

def mem_snapshot():
    p = psutil.Process(os.getpid())
    return psutil.cpu_percent(interval=None), p.memory_info().rss / (1024**2)

def mps_available():
    return torch.backends.mps.is_available() and torch.backends.mps.is_built()

# ----------------------------------------------------------------------
# 4. Ray remote tasks – each computes ONE row-block of C = A @ B
#
#    C[i] = A_rows[i] @ B       (embarrassingly parallel: no dependency
#                                 between row-blocks)
# ----------------------------------------------------------------------
@ray.remote(num_cpus=1)
def cpu_block_matmul(a_block, b):
    """C_block = A_block @ B  using NumPy (Apple Accelerate BLAS)."""
    t0 = time.time()
    c_block = a_block.dot(b)
    return c_block, time.time() - t0

@ray.remote(resources={"MPS": 1.0})
def mps_block_matmul(a_block, b):
    """C_block = A_block @ B  using PyTorch MPS (Metal GPU)."""
    device = torch.device("mps")
    a_block = np.copy(a_block)          # Ray objects are read-only
    b = np.copy(b)
    a_t = torch.from_numpy(a_block).to(device)
    b_t = torch.from_numpy(b).to(device)
    torch.mps.synchronize()

    t0 = time.time()
    c_t = torch.matmul(a_t, b_t)
    torch.mps.synchronize()             # wait for GPU kernel to finish
    elapsed = time.time() - t0

    c_block = c_t.cpu().numpy()
    del a_t, b_t, c_t
    torch.mps.empty_cache()
    return c_block, elapsed

# ----------------------------------------------------------------------
# 5. Main experiment
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print(f"PyTorch MPS available: {mps_available()}")

    num_cores = psutil.cpu_count(logical=False) or 4
    ray.init(
        num_cpus=num_cores,
        object_store_memory=OBJECT_STORE_MB * 1024**2,
        resources={"MPS": 1.0},
        ignore_reinit_error=True,
    )

    # ── Generate matrices ──────────────────────────────────────────
    A = np.random.randn(N, N).astype(np.float32)
    B = np.random.randn(N, N).astype(np.float32)
    block_rows = N // NUM_BLOCKS

    print(f"Matrix size : {N}×{N} float32  ({A.nbytes / 1e6:.1f} MB each)")
    print(f"Row-blocks  : {NUM_BLOCKS} × ({block_rows}×{N})")
    print(f"CPU cores   : {num_cores}")
    print(f"Total FLOPs : {2 * N**3 / 1e9:.1f} GFLOP")

    # Split A into row-blocks → each can be multiplied with B independently
    a_blocks = [A[i * block_rows:(i + 1) * block_rows] for i in range(NUM_BLOCKS)]
    a_refs   = [ray.put(blk) for blk in a_blocks]
    b_ref    = ray.put(B)

    # Reference result for correctness verification
    C_ref = A.dot(B)

    # ── Warm-up ────────────────────────────────────────────────────
    print("\n--- Warm-up ---")
    _ = ray.get(cpu_block_matmul.remote(a_refs[0], b_ref))
    if mps_available():
        _ = ray.get(mps_block_matmul.remote(a_refs[0], b_ref))
    print("Warm-up complete.\n")

    # ── CPU-only ───────────────────────────────────────────────────
    # All NUM_BLOCKS blocks dispatched to CPU workers.
    # Ray schedules up to `num_cores` in parallel.
    print(f"--- CPU-only ({NUM_BLOCKS} blocks across {num_cores} CPU cores) ---")
    st = time.time()
    futs = [cpu_block_matmul.remote(a_refs[i], b_ref) for i in range(NUM_BLOCKS)]
    results = ray.get(futs)
    cpu_wall = time.time() - st

    C_cpu = np.vstack([r[0] for r in results])
    cpu_times = [r[1] for r in results]
    cpu_err = float(np.max(np.abs(C_cpu - C_ref)))
    print(f"Wall : {human(cpu_wall)} s  |  Avg block: {human(np.mean(cpu_times))} s  |  Max err: {cpu_err:.2e}")

    # ── MPS-only ───────────────────────────────────────────────────
    # All blocks go through the single GPU.  Because MPS resource = 1.0,
    # Ray queues them and they execute sequentially on the GPU.
    if mps_available():
        print(f"\n--- MPS-only ({NUM_BLOCKS} blocks, sequential on GPU) ---")
        torch.mps.empty_cache()
        st = time.time()
        futs = [mps_block_matmul.remote(a_refs[i], b_ref) for i in range(NUM_BLOCKS)]
        results = ray.get(futs)
        mps_wall = time.time() - st

        C_mps = np.vstack([r[0] for r in results])
        mps_times = [r[1] for r in results]
        mps_err = float(np.max(np.abs(C_mps - C_ref)))
        print(f"Wall : {human(mps_wall)} s  |  Avg block: {human(np.mean(mps_times))} s  |  Max err: {mps_err:.2e}")
    else:
        print("\n--- MPS-only skipped (MPS not available) ---")
        mps_wall = float('inf')

    # ── Heterogeneous ──────────────────────────────────────────────
    # Interleave blocks between CPU and MPS.  The GPU processes its
    # blocks sequentially while CPU workers handle theirs in parallel.
    # Wall time ≈ max(CPU share, GPU share) — overlap is the key win.
    print(f"\n--- Heterogeneous (CPU + MPS concurrent) ---")
    if mps_available():
        torch.mps.empty_cache()
    st = time.time()
    futs   = []
    labels = []
    for i in range(NUM_BLOCKS):
        if mps_available() and i % 2 == 0:          # even → GPU
            futs.append(mps_block_matmul.remote(a_refs[i], b_ref))
            labels.append("MPS")
        else:                                        # odd  → CPU
            futs.append(cpu_block_matmul.remote(a_refs[i], b_ref))
            labels.append("CPU")
    results = ray.get(futs)
    hetero_wall = time.time() - st

    C_het = np.vstack([r[0] for r in results])
    hetero_err = float(np.max(np.abs(C_het - C_ref)))
    print(f"Wall : {human(hetero_wall)} s  |  Max err: {hetero_err:.2e}")
    for i, (r, lbl) in enumerate(zip(results, labels)):
        print(f"  Block {i:2d} ({lbl}): {human(r[1])} s")

    # ── Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"{'Mode':<18} {'Wall (s)':>10} {'Speedup vs CPU':>20}")
    print("-" * 60)
    print(f"{'CPU-only':<18} {human(cpu_wall):>10} {'—':>20}")
    if mps_wall != float('inf'):
        print(f"{'MPS-only':<18} {human(mps_wall):>10} {human(cpu_wall / mps_wall):>19}x")
    else:
        print(f"{'MPS-only':<18} {'N/A':>10} {'N/A':>20}")
    print(f"{'Heterogeneous':<18} {human(hetero_wall):>10} {human(cpu_wall / hetero_wall):>19}x")
    print("=" * 60)

    # ── Plot ───────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        os.makedirs('results/Task1', exist_ok=True)

        modes  = ['CPU-only', 'MPS-only', 'Hetero']
        times  = [cpu_wall, mps_wall if mps_wall != float('inf') else 0, hetero_wall]
        colors = ['tab:blue', 'tab:orange', 'tab:green']

        fig, ax = plt.subplots(figsize=(7, 4.5))
        bars = ax.bar(modes, times, color=colors)
        ax.set_ylabel('Wall Time (s)')
        ax.set_title(f'M1 Air 8 GB – Tiled MatMul {N}×{N} ({NUM_BLOCKS} blocks)')
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + max(times) * 0.02,
                        f'{h:.2f}s', ha='center', fontsize=10)
        plt.tight_layout()
        plt.savefig('results/Task1/m1_Matrix_multiplication_benchmark.png',
                    dpi=150, bbox_inches='tight')
        plt.show()
    except Exception as e:
        print(f"Plot skipped: {e}")

    # ── Cleanup ────────────────────────────────────────────────────
    print(f"\nFinal: CPU {mem_snapshot()[0]}%, RSS {mem_snapshot()[1]:.1f} MB")
    if mps_available():
        torch.mps.empty_cache()
    gc.collect()
    ray.shutdown()
    print("Done.")
