import os
import time
import shutil
import psutil
import numpy as np
import ray
import torch
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. Clean old Ray sessions
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
                except Exception as e:
                    print(f"Failed to clean {item}: {e}")
cleanup_ray_tmp()

# ----------------------------------------------------------------------
# 2. Config – safe for 8 GB M1 Air
# ----------------------------------------------------------------------
WIDTH, HEIGHT = 3000, 3000
MAX_ITER = 256
TILE_SIZE = 750                    # 750×750 → ~4.5 MB per tile
OBJECT_STORE_MB = 400

# ----------------------------------------------------------------------
# 3. Pixel-perfect tile coordinate mapping
# ----------------------------------------------------------------------
def get_tile_coords(tile_x, tile_y, tile_w, tile_h, total_w, total_h):
    x0, x1 = -2.0, 1.0
    y0, y1 = -1.5, 1.5

    pixel_x0 = tile_x * tile_w
    pixel_x1 = min((tile_x + 1) * tile_w, total_w)
    pixel_y0 = tile_y * tile_h
    pixel_y1 = min((tile_y + 1) * tile_h, total_h)

    actual_w = pixel_x1 - pixel_x0
    actual_h = pixel_y1 - pixel_y0

    real_x0 = x0 + (pixel_x0 / total_w) * (x1 - x0)
    real_x1 = x0 + (pixel_x1 / total_w) * (x1 - x0)
    real_y0 = y0 + (pixel_y0 / total_h) * (y1 - y0)
    real_y1 = y0 + (pixel_y1 / total_h) * (y1 - y0)

    return real_x0, real_x1, real_y0, real_y1, actual_w, actual_h

# ----------------------------------------------------------------------
# 4. Ray setup
# ----------------------------------------------------------------------
def setup_ray():
    ray.init(
        num_cpus=psutil.cpu_count(logical=False),
        object_store_memory=OBJECT_STORE_MB * 1024**2,
        resources={"MPS": 1.0},
        ignore_reinit_error=True,
        log_to_driver=False,
    )
    print(f"Ray started – rendering {WIDTH}×{HEIGHT} Mandelbrot in {TILE_SIZE}×{TILE_SIZE} tiles")

# ----------------------------------------------------------------------
# 5. Mandelbrot kernel – CHUNKED & MPS-SAFE
# ----------------------------------------------------------------------
def mandelbrot_tile_torch(x0, x1, y0, y1, width, height, max_iter, device):
    if width <= 0 or height <= 0:
        return np.full((height, width), max_iter, dtype=np.int32)

    x = torch.linspace(x0, x1, width, device=device, dtype=torch.float32)
    y = torch.linspace(y0, y1, height, device=device, dtype=torch.float32)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    c = X + 1j * Y
    z = torch.zeros_like(c)
    iter_count = torch.full(c.shape, max_iter, dtype=torch.int32, device=device)
    active = torch.ones(c.shape, dtype=torch.bool, device=device)

    for i in range(max_iter):
        if not torch.any(active):
            break

        z_active = z[active]
        c_active = c[active]
        z_active = z_active * z_active + c_active
        z[active] = z_active

        abs_z = torch.abs(z_active)
        escaped = abs_z > 2
        iter_count[active] = torch.where(escaped, torch.full_like(iter_count[active], i), iter_count[active])
        active[active.clone()] = ~escaped

        del z_active, c_active, abs_z, escaped
        if device.type == 'mps':
            torch.mps.empty_cache()

    return iter_count.cpu().numpy()

# ----------------------------------------------------------------------
# 6. Ray tasks – tile-based
# ----------------------------------------------------------------------
@ray.remote(num_cpus=1)
def render_tile_cpu(x0, x1, y0, y1, w, h):
    device = torch.device("cpu")
    start = time.time()
    tile = mandelbrot_tile_torch(x0, x1, y0, y1, w, h, MAX_ITER, device)
    return tile, time.time() - start

@ray.remote(resources={"MPS": 1.0})
def render_tile_mps(x0, x1, y0, y1, w, h):
    device = torch.device("mps")
    torch.mps.empty_cache()
    start = time.time()
    tile = mandelbrot_tile_torch(x0, x1, y0, y1, w, h, MAX_ITER, device)
    return tile, time.time() - start

# ----------------------------------------------------------------------
# 7. Unified tile task creator – supports cpu/mps/hetero
# ----------------------------------------------------------------------
def create_tile_tasks(grid_w, grid_h, mode="hetero"):
    """
    mode: 'cpu', 'mps', or 'hetero'
    """
    futures = []
    tile_info = []

    for tile_y in range(grid_h):
        for tile_x in range(grid_w):
            x0, x1, y0, y1, w, h = get_tile_coords(tile_x, tile_y, TILE_SIZE, TILE_SIZE, WIDTH, HEIGHT)
            tile_idx = tile_y * grid_w + tile_x

            if mode == "cpu":
                fut = render_tile_cpu.remote(x0, x1, y0, y1, w, h)
            elif mode == "mps":
                fut = render_tile_mps.remote(x0, x1, y0, y1, w, h)
            elif mode == "hetero":
                if tile_idx % 2 == 0:
                    fut = render_tile_mps.remote(x0, x1, y0, y1, w, h)
                else:
                    fut = render_tile_cpu.remote(x0, x1, y0, y1, w, h)
            else:
                raise ValueError("mode must be 'cpu', 'mps', or 'hetero'")

            futures.append(fut)
            tile_info.append((tile_y, tile_x, w, h))

    return futures, tile_info

# ----------------------------------------------------------------------
# 8. Image assembly from tiles
# ----------------------------------------------------------------------
def assemble_image(tiles, tile_info, total_h, total_w):
    img = np.zeros((total_h, total_w), dtype=np.int32)
    for (tile_data, _), (tile_y, tile_x, w, h) in zip(tiles, tile_info):
        y0 = tile_y * TILE_SIZE
        x0 = tile_x * TILE_SIZE
        img[y0:y0+h, x0:x0+w] = tile_data[:h, :w]
    return img

# ----------------------------------------------------------------------
# 9. Save image
# ----------------------------------------------------------------------
def save_image(img, filename, title):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.figure(figsize=(6,6), dpi=150)
    plt.imshow(img, cmap='magma', extent=[-2, 1, -1.5, 1.5], origin='lower')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved: {filename}")

# ----------------------------------------------------------------------
# 10. Main experiment
# ----------------------------------------------------------------------
def run():
    mps_available = torch.backends.mps.is_available()
    if mps_available:
        print("MPS backend ready")
    else:
        print("MPS not available – GPU runs will be skipped.")

    setup_ray()

    grid_w = (WIDTH + TILE_SIZE - 1) // TILE_SIZE
    grid_h = (HEIGHT + TILE_SIZE - 1) // TILE_SIZE
    total_tiles = grid_w * grid_h
    print(f"Splitting into {grid_w}×{grid_h} = {total_tiles} tiles")

    results = {}

    # ------------------------------------------------------------------
    # CPU-only
    # ------------------------------------------------------------------
    print("\n--- CPU-only (all tiles on CPU) ---")
    st = time.time()
    cpu_futs, tile_info = create_tile_tasks(grid_w, grid_h, mode="cpu")
    cpu_results = ray.get(cpu_futs)
    cpu_wall_time = time.time() - st
    cpu_img = assemble_image(cpu_results, tile_info, HEIGHT, WIDTH)
    save_image(cpu_img, "results/Task3/mandelbrot_cpu.png", "Mandelbrot – CPU")
    results["CPU-only"] = cpu_wall_time

    # ------------------------------------------------------------------
    # MPS-only (ALL tiles on GPU)
    # ------------------------------------------------------------------
    if mps_available:
        print("\n--- MPS-only (ALL tiles on GPU) ---")
        torch.mps.empty_cache()
        st = time.time()
        mps_futs, tile_info = create_tile_tasks(grid_w, grid_h, mode="mps")
        mps_results = ray.get(mps_futs)
        mps_wall_time = time.time() - st
        mps_img = assemble_image(mps_results, tile_info, HEIGHT, WIDTH)
        save_image(mps_img, "results/Task3/mandelbrot_mps.png", "Mandelbrot – MPS")
        results["MPS-only"] = mps_wall_time
    else:
        print("\n--- MPS-only skipped (MPS not available) ---")
        results["MPS-only"] = float('inf')

    # ------------------------------------------------------------------
    # Heterogeneous
    # ------------------------------------------------------------------
    print("\n--- Heterogeneous (CPU + MPS in parallel) ---")
    if mps_available:
        torch.mps.empty_cache()
    st = time.time()
    hetero_futs, tile_info = create_tile_tasks(grid_w, grid_h, mode="hetero")
    hetero_results = ray.get(hetero_futs)
    hetero_wall_time = time.time() - st
    hetero_img = assemble_image(hetero_results, tile_info, HEIGHT, WIDTH)
    save_image(hetero_img, "results/Task3/mandelbrot_hetero.png", "Mandelbrot – Hetero")
    results["Hetero"] = hetero_wall_time

    # ------------------------------------------------------------------
    # Summary + Plot
    # ------------------------------------------------------------------
    print("\n" + "="*70)
    print(f"{'Mode':<12} {'Wall (s)':>10} {'Speedup vs CPU':>18}")
    print("-"*70)
    cpu_time = results["CPU-only"]
    for mode in ["CPU-only", "MPS-only", "Hetero"]:
        t = results[mode]
        if t == float('inf'):
            speedup = "N/A"
        else:
            speedup = f"{cpu_time/t:.2f}x" if t > 0 else "inf"
        print(f"{mode:<12} {t:>10.3f} {speedup:>18}")
    print("="*70)

    # Bar chart
    modes = ['CPU', 'MPS', 'Hetero']
    times = [results["CPU-only"], results.get("MPS-only", 0), results["Hetero"]]
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    if not mps_available:
        times[1] = 0
        colors[1] = 'lightgray'

    plt.figure(figsize=(7,4))
    bars = plt.bar(modes, times, color=colors)
    plt.ylabel('Wall Time (s)')
    plt.title('M1 Air 8 GB – Mandelbrot 3000×3000 (Tiled)')
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            plt.text(bar.get_x() + bar.get_width()/2, h + max(times)*0.02, f'{h:.2f}s', ha='center', fontsize=10)
    plt.ylim(0, max(t for t in times if t > 0)*1.3)
    plt.tight_layout()
    plt.savefig('results/Task3/mandelbrot_timing.png', dpi=150)
    plt.show()

    ray.shutdown()
    print("\nAll done. Images saved in results/Task3/")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    run()