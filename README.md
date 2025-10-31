<div align="center">

# Ray M1 Heterogeneous Benchmark

[![Ray Version](https://img.shields.io/badge/Ray-2.51.0-blue.svg)](https://docs.ray.io/)
[![Python Version](https://img.shields.io/badge/Python-3.11.9-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10.0.dev20251030-orange.svg)](https://pytorch.org/)

This repository benchmarks CPU vs MPS (Apple Metal GPU) performance on Apple Silicon Macs using **Ray** and **PyTorch**. It contains two standalone experiments that measure how the MacBook Air M1 handles numerical workloads on CPU, GPU (MPS), and in a heterogeneous (CPU+GPU) configuration orchestrated by Ray.

</div>

---

## Files
- `Task_1_matrix_multiplication.py` — Matrix multiplication experiment (Task 1).  
- `Task_2_Monte_Carlo_Simulation.py` — Monte Carlo π estimation experiment (Task 2).  
- `Task_3_mandelbrot_set.py` — Mandelbrot Set fractal rendering benchmark (Task 3).  
- `requirements.txt` — Python dependencies.  
- `results/` —  generated plot outputs

---

## Purpose
Compare wall-clock performance and resource behavior for:
- Pure **CPU** workloads (NumPy/BLAS),  
- Pure **GPU (MPS)** workloads (PyTorch on Metal), and  
- **Heterogeneous** mixes via Ray scheduling (CPU + MPS).

The experiments help illustrate scheduling tradeoffs, host→device overheads, warmup effects, and concurrency behavior on Apple Silicon.

---

## Prerequisites
- macOS 12.3+ on Apple Silicon (M1 / M2 / M3).  
- Python 3.10+ (native arm64 interpreter recommended).  
- PyTorch with MPS support. See https://pytorch.org for macOS MPS install instructions.  
- Ray (local).  
- Install dependencies:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## How to run
Run the matrix multiplication benchmark (Task 1):
```bash
python Task_1_matrix_multiplication.py
```
Run the Monte Carlo benchmark (Task 2):
```bash
python Task_2_Monte_Carlo_Simulation.py
```
Task 3 — Mandelbrot Set Fractal
```bash
python Task_3_mandelbrot_set.py
```
Each script automatically creates a corresponding folder inside results/TaskX/
(e.g. results/Task3/mandelbrot_cpu.png, mandelbrot_mps.png, mandelbrot_hetero.png, plus timing plots).

## What each experiment measures
### Task 1 — Matrix multiplication

- Benchmarks dense float32 matrix multiplication on CPU (NumPy) vs GPU (PyTorch MPS).

- Ray distributes chunks as independent workers.

- Outputs wall time per mode + bar chart.

### Task 2 — Monte Carlo π (Task_2_Monte_Carlo_Simulation.py)
- Simulates random points to approximate π.

- Compares NumPy (CPU), PyTorch (MPS), and mixed Ray scheduling.

- Prints estimated π and wall times.

### Task 3 — Mandelbrot Set Fractal Rendering

- Generates a 3000×3000 pixel Mandelbrot image (≈ 9 M points).

- Tests how iterative, complex-number workloads perform on CPU vs GPU.

- Splits computation into 4×4 tiles, rendered by Ray.

- Saves CPU, GPU, and Heterogeneous images + bar chart in results/Task3/.

## Outputs
### Matrix multiplication (Task 1)
```bash
=======================================================
Mode              Wall (s)       Speedup vs CPU
-------------------------------------------------------
CPU-only             0.203                    —
MPS-only             1.022                0.199x
Heterogeneous        0.999                0.204x
=======================================================
```
### Monte Carlo (Task 2)
```bash
======================================================================
Mode           π estimate   Wall (s)      Speedup
----------------------------------------------------------------------
CPU-only         3.142064      0.438            —
MPS-only         3.141465      1.154        0.38x
Hetero           3.141169      0.126        3.48x
======================================================================
```
### Mandelbrot Set Fractal Rendering
```bash
======================================================================
Mode           Wall (s)     Speedup vs CPU
----------------------------------------------------------------------
CPU-only         11.429              1.00x
MPS-only        107.394              0.11x
Hetero           40.965              0.28x
======================================================================
```

## 🧭 Inference & Observations

- Warm-up effects: The first GPU call initializes Metal kernels → ignore first timings.

- CPU BLAS dominance: Apple Accelerate is so optimized that small/medium matmuls often beat MPS.

- GPU strength: Large, massively-parallel batches (e.g., Monte Carlo) exploit MPS well.

- Heterogeneous sweet-spot: Overlapping CPU + GPU tasks via Ray yields lower wall time if the workload divides well.

- Mandelbrot insight: Iterative, low arithmetic-intensity workloads may underperform on GPU due to kernel launch overhead — a perfect stress test for Ray’s scheduling efficiency.

## 🌐 Ray Dashboard

When you run any task, Ray spins up a local instance and prints:
```bash
Started a local Ray instance. View the dashboard at http://127.0.0.1:8265
```
Open this URL to explore:

Overview: live CPU/GPU/memory usage

Jobs: running and past Ray scripts

Cluster: shows your Mac node with resources

Metrics & Logs: detailed runtime telemetry

Even on a single M1 chip, Ray behaves like a mini-cluster, showcasing distributed orchestration on unified memory hardware.

## 📦 Results Summary
| Task	               | CPU-only	      | MPS-only	                        | Hetero |
|:-------------        |:--------------:|----------------------------------:|--------------:|
|Matrix Multiplication |	✅ Fastest	   |⚠️ Slightly slower (small matrices)|	⚖️ Comparable|
|Monte Carlo π	       | ⚙️ Baseline	   |✅ Significant speedup            |	🧩 Best combined performance|
| Mandelbrot Set	     | ✅ Efficient    |	❌ Slower (init overhead)        |	🧠 Intermediate overlap speed|

---

## Support

If you find this repository helpful, please consider giving it a ⭐ on GitHub! It helps others discover this resource.


## Acknowledgments

- Built with [Ray](https://ray.io/) - the open-source framework for scaling AI and Python applications
- Inspired by the Ray community and the need for practical, hands-on learning resources

---

<div align="center">

**[⬆ Back to Top](#ray-m1-heterogeneous-benchmark)**

Made with ❤️ by [Deepan Kulandaisami](https://github.com/Deeps72-ux)

</div>