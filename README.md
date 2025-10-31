# Ray M1 Heterogeneous Benchmark

This repository benchmarks CPU vs MPS (Apple Metal GPU) performance on Apple Silicon Macs using **Ray** and **PyTorch**. It contains two standalone experiments that measure how the MacBook Air M1 handles numerical workloads on CPU, GPU (MPS), and in a heterogeneous (CPU+GPU) configuration orchestrated by Ray.

---

## Files
- `Task_1_matrix_multiplication.py` — Matrix multiplication experiment (Task 1).  
- `Task_2_Monte_Carlo_Simulation.py` — Monte Carlo π estimation experiment (Task 2).  
- `requirements.txt` — Python dependencies.  
- `results` —  generated plot outputs

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

## Environment tuning (recommended for fair comparisons)
Set BLAS / thread controls before importing NumPy/PyTorch to avoid hidden multi-thread contention:

## Parameters
N — matrix size (Task 1).

NUM_CPU_TASKS — number of concurrent CPU tasks.

NUM_MPS_TASKS — number of concurrent MPS tasks.

TOTAL_SAMPLES, BATCH — Monte Carlo totals and per-task batch size (Task 2).

object_store_memory — Ray object store size (tune down on low-memory machines).

## What each experiment measures
Task 1 — Matrix multiplication (ray_m1_hetero_benchmark.py)

Creates two large float32 matrices and places them in Ray’s object store.

Runs:

CPU-only parallel NumPy .dot() tasks,

MPS-only PyTorch matmul tasks (explicitly converting arrays to device),

Mixed (heterogeneous) run of CPU + MPS tasks.

Reports per-task times, wall times, CPU%, RSS, and saves a bar chart (if matplotlib available).

Task 2 — Monte Carlo π (Task_2_Monte_Carlo_Simulation.py)

Splits TOTAL_SAMPLES into BATCH sized tasks.

Runs:

CPU-only Monte Carlo using NumPy,

MPS-only Monte Carlo using PyTorch on device='mps',

Mixed run alternating CPU and GPU tasks.

Reports π estimates and wall times for each mode.

## Outputs
### Matrix multiplication (Task 1)
```bash
PyTorch MPS available: True
--- Warm-up ---
--- CPU-only (4 tasks) ---
Wall: 0.245 s | Per-task: 0.054
--- MPS-only (1 task) ---
Wall: 1.110 s | Time: 0.041 s
--- Heterogeneous (CPU 4 + MPS 1) ---
Wall: 1.239 s
  CPU 0: 0.096 s
  ...
  MPS 0: 0.226 s
```
### Monte Carlo (Task 2)
```bash
Running 10 batches of 2000000 samples each (20000000 total)
Results:
 CPU-only  : π ≈ 3.142168  time = 0.382 s
 GPU-only  : π ≈ 3.141850  time = 1.450 s
 Hetero    : π ≈ 3.141747  time = 0.104 s

```

## Inference
Warmup matters: MPS/PyTorch may include one-time kernel setup cost — discard first-run timings for fairness.

CPU BLAS is highly optimized: small-to-medium dense matmuls may run faster on CPU due to Apple Accelerate.

GPU wins for massively parallel work: embarrassingly parallel tasks with large batch sizes favor the GPU.

Heterogeneous runs: combining CPU+GPU with Ray can yield the best wall time by overlapping work, but profile per-device to understand contributions.

