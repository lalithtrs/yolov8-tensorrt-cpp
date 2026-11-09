# Reproducing the results

There are three ways to check this project, from no effort to a full re-run:

| Option | What you need | Time | What you get |
|---|---|---|---|
| **A. Read the executed notebook** | a browser | minutes | every output of the reference run |
| **B. Re-run on Kaggle** | a free Kaggle account | ~1-1.5 h | a new run on the same GPU type (T4) |
| **C. Run on your own PC / laptop** | Linux or WSL2 + NVIDIA GPU | ~1-2 h | a run on your GPU, compared against ours |

## Is it reproducible?

**The method is fully reproducible; absolute speed depends on the hardware.**

Fixed in the code, identical in every run:
- model: official Ultralytics `yolov8s.pt`, exported to ONNX with fixed settings (640x640, batch 1, opset 17)
- data: COCO val2017, sorted by file name; first 4500 images for mAP, first 300 for latency, last 500 for
  INT8 calibration (never scored)
- preprocessing, thresholds (conf 0.25 / NMS 0.45 for speed; conf 0.001 / NMS 0.7 / max 300 for mAP),
  warm-up (50 frames), timed frames (500), repeats (3, median)
- package versions: `requirements.txt` + TensorRT 10.16.1.11 (installed automatically)

Can vary between runs:

| Source | Effect | What to expect |
|---|---|---|
| Different GPU | absolute latency changes completely | compare **speedups** and **mAP**, not milliseconds |
| Same GPU type, new engine build | TensorRT picks kernels by timing on the build machine, so two builds can differ slightly | small latency differences; FP32/FP16 mAP should stay within a few tenths of a point |
| INT8 calibration | scales come from the calibration images and the build | INT8 mAP may move more than FP16; the tolerance used below is 1.0 point |
| GPU power and clocks | the T4 is capped at 70 W and lowers its clock when kept busy (see finding 10 in `FINDINGS.md`); laptops also throttle on heat and battery | a few percent run to run; keep the machine on AC power and idle |
| CPU | pre/post-processing run on the CPU | the CPU columns depend on your processor |
| Newer Kaggle images | CUDA / PyTorch versions change over time | the exact TensorRT pin may not install; the notebook then uses the latest 10.x and records the version |

How stable the reference run was: the spread of total latency across the 3 repeats was 0.01-0.76 ms, and
separate 2000-frame runs reproduced the medians within 0.02 ms (6.60 vs 6.61 ms, 4.27 vs 4.25 ms). Engines
were built once, so build-to-build variation was not measured.

`scripts/compare_results.py` checks a new run against the reference: mAP within 0.5 points (1.0 for INT8)
counts as reproduced; latency is compared in ms on the same GPU type and as speedups otherwise.

## A. Read the executed notebook

Open the public Kaggle notebook: https://www.kaggle.com/code/lalithtrs/yolo-trt-kaggle
(a copy is saved as `notebooks/yolo_trt_kaggle_run.ipynb`). It is the
reference run with every output saved: environment, build log, benchmarks, mAP, tables, charts and the demo.
`notebooks/yolo_trt_kaggle.ipynb` is the same pipeline without outputs, kept in sync with the code;
use that one to re-run.

## B. Re-run on Kaggle

1. Either open https://www.kaggle.com/code/lalithtrs/yolo-trt-kaggle and click **Copy & Edit**, or Kaggle -> **Create -> Import Notebook** ->
   upload `notebooks/yolo_trt_kaggle.ipynb`. The uploaded file has the latest code (version pins,
   fixes and the MinMax / clock steps as built-in cells); the public notebook is the exact reference run.
2. Settings: **Accelerator: GPU T4** (x1 or x2), **Internet: On** (phone-verified account).
3. Optional: **Add Input** -> a COCO 2017 dataset (saves the ~1 GB download).
4. **Run All**. `QUICK = True` in the first cell gives a ~20 min smoke run.
5. Download `yolov8-tensorrt-cpp-results.zip` from the Output panel, unzip, then compare:
   ```bash
   python scripts/compare_results.py results <unzipped>/yolov8-tensorrt-cpp/results
   ```

## C. Run on your own PC / laptop (GPU supported)

### Requirements

| | Minimum | Notes |
|---|---|---|
| OS | Ubuntu 22.04 / 24.04, or **Windows 11 + WSL2** (Ubuntu) | native Windows is not supported (the scripts assume Linux paths and `.so` libraries) |
| GPU | NVIDIA, **4 GB+ VRAM**, RTX 20-series / T4 or newer recommended | INT8 and FP16 speedups need Tensor Cores (RTX 20xx and newer) |
| Driver | `nvidia-smi` shows **CUDA Version >= 12.6** | on WSL2, install the normal Windows NVIDIA driver; do not install a Linux driver inside WSL |
| Python | 3.10-3.13 | TensorRT has no wheels for 3.14 yet |
| Disk / RAM | ~6 GB free, 8 GB RAM | COCO val2017 is ~1 GB; engines ~110 MB |

You do **not** need to install the CUDA Toolkit or TensorRT by hand: PyTorch and TensorRT come from pip,
and `scripts/setup_cpp.py` finds their libraries and fetches the matching C++ headers.

### Step 1: get the code

```bash
git clone https://github.com/<your-user>/yolov8-tensorrt-cpp.git
cd yolov8-tensorrt-cpp
```

### Step 2: system packages (once)

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git pkg-config wget unzip python3-venv libopencv-dev
nvidia-smi        # must list your GPU and "CUDA Version: 12.6" or higher
```

If your default `python3` is 3.14 or newer, also install an older one, e.g. `sudo apt-get install -y python3.12 python3.12-venv`
(on Ubuntu 22.04 via the deadsnakes PPA).

### Step 3: Python environment, PyTorch and TensorRT (once)

```bash
bash scripts/install_local.sh
# or, to choose the Python version:
PYTHON=python3.12 bash scripts/install_local.sh
```

It checks the driver and system packages, creates `.venv`, installs PyTorch built for your driver's CUDA
version, the pinned packages from `requirements.txt` and TensorRT 10.16.1.11, then confirms PyTorch can see
the GPU. Every check stops with the exact command to fix it.

### Step 4: quick smoke run (~20 min)

```bash
source .venv/bin/activate
python scripts/run_all.py --quick
```

The first two steps are the risky ones: `build` compiles the C++ app and runs `hello_trt`, which must print
`OK: TensorRT runtime created from C++`. If that passes, the rest is routine.

### Step 5: full run (~1-2 h, depending on the GPU)

```bash
python scripts/run_all.py
# COCO already on disk (needs val2017/ and annotations/instances_val2017.json):
python scripts/run_all.py --coco-dir /path/to/coco
```

Everything goes to `runs/local/` (data lists, model, engines, results), so the reference `results/` is never
overwritten. A failed step can be resumed, for example `--steps coco,minmax,clocks,results,demo`. Existing
engines are reused unless you pass `--rebuild-engines`.

| Step | What it does | Typical time on a T4 |
|---|---|---|
| `check` | GPU, driver, versions -> `env.json` | seconds |
| `build` | C++ app + TensorRT smoke test | 1-2 min |
| `data` | COCO val2017 + image lists | 2-5 min (download) |
| `onnx` | export YOLOv8s | < 1 min |
| `engines` | FP32 / FP16 / INT8 engines | ~20 min (INT8 ~13 min) |
| `pytorch` | PyTorch latency + mAP | 10-15 min |
| `bench` | TensorRT latency, baseline vs optimised CPU path | 3-5 min |
| `coco` | TensorRT mAP | 5-10 min |
| `minmax` | INT8 MinMax experiment | ~15 min |
| `clocks` | GPU clock / power log | 1 min |
| `results` | tables, checks, charts | seconds |
| `demo` | demo video + GIF | < 1 min |

### Step 6: compare with the reference run

```bash
python scripts/compare_results.py results runs/local/results --out runs/local/results/comparison.md
```

It prints mAP per variant (reproduced or not) and latency with speedups. On a different GPU the absolute
milliseconds will differ; the speedups and the pattern (FP16 >> FP32, CPU stages become the bottleneck,
MinMax beats entropy for INT8) are what should hold.

### Tips for laptops

- Plug in the charger and pick the high-performance power mode; on battery the GPU clocks drop a lot.
- Close games, browsers with video, and other GPU apps; check with `nvidia-smi` that nothing else is running.
- Laptops heat up during the ~1 h run; the `clocks` step shows whether your GPU throttled.
- Hybrid graphics (Intel/AMD + NVIDIA): `nvidia-smi` must list the NVIDIA GPU; use `--gpu 1` if it is not GPU 0.

### Troubleshooting

| Problem | Fix |
|---|---|
| `nvidia-smi not found` / no GPU in WSL2 | update the Windows NVIDIA driver; run `wsl --update`; restart WSL |
| `driver CUDA ... too old` | update the NVIDIA driver until `nvidia-smi` shows CUDA >= 12.6 |
| `tensorrt-cuXX==10.16.1.11 not available` | the installer falls back to the latest 10.x automatically; results stay comparable |
| `cmake configure failed` | `sudo apt-get install libopencv-dev`; check `pkg-config --modversion opencv4` |
| `hello_trt` fails with a version mismatch | delete `/tmp/trt_oss` and re-run `--steps build` (re-fetches headers) |
| Engine load fails in C++ | engines are tied to GPU + TensorRT version: `--rebuild-engines` |
| Out of GPU memory while building | close other GPU apps; 4 GB is the practical minimum |
| COCO download fails | download `val2017.zip` and `annotations_trainval2017.zip` from cocodataset.org, unzip into one folder, pass `--coco-dir` |
| Runs much slower than expected | laptop on battery or hot: see the tips above and the `clocks` chart |
