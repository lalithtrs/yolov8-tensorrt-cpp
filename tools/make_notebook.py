#!/usr/bin/env python3
"""Generate notebooks/yolo_trt_kaggle.ipynb from the source files in this repo.

The notebook writes every source file with %%writefile, so it runs on Kaggle with no GitHub access.
Re-run this script after editing any file:  python3 tools/make_notebook.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KREPO = '/kaggle/working/yolov8-tensorrt-cpp'
FILES = ['CMakeLists.txt', 'src/hello_trt.cpp', 'src/main.cpp', 'scripts/coco_eval.py', 'scripts/setup_cpp.py',
         'scripts/collect_env.py', 'scripts/prepare_data.py', 'scripts/export_onnx.py', 'scripts/build_engine.py',
         'scripts/pytorch_baseline.py', 'scripts/run_trt.py', 'scripts/make_results.py', 'scripts/make_demo.py', 'scripts/plot_clocks.py',
         'requirements.txt', 'README.md', '.gitignore']

cells = []


def lines(text):
    text = text.strip('\n')
    ls = text.split('\n')
    return [l + '\n' for l in ls[:-1]] + [ls[-1]]


def md(text):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': lines(text)})


def code(text, hidden=False):
    meta = {'jupyter': {'source_hidden': True}} if hidden else {}
    cells.append({'cell_type': 'code', 'metadata': meta, 'execution_count': None, 'outputs': [],
                  'source': lines(text)})


md("""
# YOLOv8s → TensorRT C++ pipeline: full run on Kaggle

This notebook does the whole project end to end and prints every result:

1. checks the GPU, installs TensorRT / Ultralytics / OpenCV, **builds the C++ app and runs a TensorRT smoke test first**
2. gets COCO val2017, exports YOLOv8s to ONNX, builds **FP32 / FP16 / INT8** TensorRT engines
3. measures the **PyTorch baseline** (latency + mAP)
4. measures the **C++ TensorRT pipeline** per stage (P50 / P95), baseline vs optimised CPU path, and **COCO mAP**
5. writes a results table, correctness checks, a latency chart and a demo GIF

### How to run
1. **Settings → Accelerator → GPU T4** (T4 ×2 is fine, GPU 0 is used). Not P100: Pascal has no fast FP16/INT8.
2. **Settings → Internet → On** (needs a phone-verified Kaggle account).
3. Optional: **Add Input → search "COCO 2017"** and attach a dataset with `val2017/` + `instances_val2017.json` to skip the ~1 GB download. Otherwise it is downloaded automatically.
4. **Run All.** Full run takes roughly 1–1.5 h. Set `QUICK = True` below for a ~20 min smoke run on fewer images.

The run stops at the first failing step, with the error printed under that cell.
All results land in `/kaggle/working/yolov8-tensorrt-cpp/results/` and in `yolov8-tensorrt-cpp-results.zip` (Output panel).
""")

code(f"""
import os, json, subprocess, sys

QUICK = False                      # True = fast smoke run (fewer images, 1 benchmark repeat)
REPO = '{KREPO}'
N_EVAL = 500 if QUICK else 4500    # images scored for mAP (first N of val2017)
N_CALIB = 200 if QUICK else 500    # INT8 calibration images (last N of val2017, never scored)
REPEATS = 1 if QUICK else 3        # benchmark repeats (median is reported)
ITERS = 200 if QUICK else 500      # timed frames per benchmark (after 50 warm-up frames)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'   # same single GPU for every measurement
for d in ['src', 'scripts', 'results', 'data', 'models', 'engines']:
    os.makedirs(f'{{REPO}}/{{d}}', exist_ok=True)

def sh(cmd):
    \"\"\"Run a shell command in the repo, stream its output, stop the notebook if it fails.\"\"\"
    print('$', cmd, flush=True)
    p = subprocess.Popen(cmd, shell=True, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end='', flush=True)
    if p.wait() != 0:
        raise RuntimeError(f'command failed (exit {{p.returncode}}): {{cmd}}')

gpu = subprocess.run('nvidia-smi --query-gpu=name --format=csv,noheader -i 0', shell=True,
                     capture_output=True, text=True).stdout.strip()
print('GPU 0:', gpu or 'NONE')
assert gpu, 'No GPU. Settings > Accelerator > GPU T4, then restart the session.'
if 'T4' not in gpu:
    print('WARNING: this project assumes a T4; numbers on other GPUs are not comparable.')
""")

md("""
## Step 1: Project source files
Every file of the repo is written here, so the notebook needs nothing but Internet access.
(Generated from the repo by `tools/make_notebook.py`; the code is identical to the GitHub version.)
""")
for rel in FILES:
    with open(os.path.join(ROOT, rel)) as f:
        body = f.read()
    code(f'%%writefile {KREPO}/{rel}\n' + body.rstrip('\n'), hidden=False)

md("""
## Step 2: Install dependencies
Pinned to the versions used for the published results (`requirements.txt`, TensorRT 10.16.1.11).
If Kaggle's CUDA no longer has that exact TensorRT build, the latest TensorRT 10.x is installed instead
and the version is recorded in `results/env.json`. OpenCV for C++ comes from apt.
""")
code("""
import torch
CU = torch.version.cuda.split('.')[0]
TRT_VERSION = '10.16.1.11'   # version used for the published results
print('torch', torch.__version__, '| CUDA', torch.version.cuda, '| GPU', torch.cuda.get_device_name(0))
sh('pip install -q -r requirements.txt')
try:
    sh(f'pip install -q "tensorrt-cu{CU}=={TRT_VERSION}"')
except RuntimeError:
    print(f'tensorrt-cu{CU}=={TRT_VERSION} not available here; installing the latest TensorRT 10.x')
    sh(f'pip install -q "tensorrt-cu{CU}>=10.0,<11"')
sh('apt-get update -qq > /dev/null && apt-get install -y -qq libopencv-dev cmake pkg-config > /dev/null && echo apt OK')
sh('python -c "import tensorrt, ultralytics; print(tensorrt.__version__, ultralytics.__version__)"')
""")

md("""
## Step 3: Build the C++ app and smoke-test TensorRT from C++
This is the riskiest part, so it runs first. `setup_cpp.py` finds `libnvinfer` from the pip package,
fetches the matching C++ headers from GitHub, finds CUDA, and builds with CMake.
If `hello_trt` prints **OK**, the C++/TensorRT/CUDA toolchain works.
""")
code("""
sh('python scripts/setup_cpp.py')
sh('./build/hello_trt')
""")

md("## Step 4: Record versions and get COCO val2017")
code("""
sh('python scripts/collect_env.py results/env.json')
sh(f'python scripts/prepare_data.py --out-dir data --n-eval {N_EVAL} --n-calib {N_CALIB}')
DATA = json.load(open(f'{REPO}/data/data.json'))
ANN = DATA['ann']
""")

md("## Step 5: Export YOLOv8s to ONNX (static 1×3×640×640, opset 17)")
code("""
sh('cd models && python ../scripts/export_onnx.py --weights yolov8s.pt --out yolov8s.onnx')
""")

md("""
## Step 6: Build the TensorRT engines (FP32, FP16, INT8)
INT8 takes longest because it runs calibration on `data/calib.txt` (images that are never scored).
Engines only work with the GPU and TensorRT version that built them, so they are rebuilt every session.
""")
code("""
for prec in ['fp32', 'fp16', 'int8']:
    extra = ' --calib-list data/calib.txt' if prec == 'int8' else ''
    sh(f'python scripts/build_engine.py models/yolov8s.onnx engines/yolov8s_{prec}.engine {prec}{extra} '
       f'--log results/engines.json')
""")

md("""
## Step 7: PyTorch baseline (Ultralytics), FP32 and FP16
Latency per stage on the benchmark images, then COCO mAP on the evaluation images.
Expect about 44–45 mAP50-95 for YOLOv8s (Ultralytics reports 44.9 on all 5000 images).
""")
code("""
sh(f'python scripts/pytorch_baseline.py --weights models/yolov8s.pt --bench-list data/val_bench.txt '
   f'--eval-list data/val_eval.txt --ann {ANN} --out-dir results --iters {ITERS}')
""")

md("""
## Step 8: TensorRT C++ latency
Each engine runs with the **baseline** CPU path (plain OpenCV preprocessing, strided decode) and the
**optimised** path (`--fast`: one fused preprocessing pass, contiguous vectorised decode).
Repeats are interleaved and the median is kept. The last part checks that both paths produce the same detections.
""")
code("""
sh(f'python scripts/run_trt.py --bin build/yolo_trt --engines-dir engines --bench-list data/val_bench.txt '
   f'--eval-list data/val_eval.txt --ann {ANN} --out-dir results --repeats {REPEATS} --iters {ITERS} --skip-coco')
""")

md("## Step 9: TensorRT C++ accuracy (COCO mAP per engine)")
code("""
sh(f'python scripts/run_trt.py --bin build/yolo_trt --engines-dir engines --bench-list data/val_bench.txt '
   f'--eval-list data/val_eval.txt --ann {ANN} --out-dir results --skip-bench')
""")

md("""
## Step 9b: INT8 calibration experiment (MinMax vs entropy)
The default INT8 engine uses entropy calibration. This builds a second INT8 engine with MinMax calibration
(keeps the full activation range instead of clipping) and measures its latency and mAP the same way.
Results go to `results/int8_minmax/` and appear as an extra row in the results table.
""")
code("""
sh('python scripts/build_engine.py models/yolov8s.onnx engines/yolov8s_int8mm.engine int8 '
   '--calib-list data/calib.txt --calibrator minmax --log results/int8_minmax_engine.json')
sh(f'python scripts/run_trt.py --bin build/yolo_trt --engines-dir engines --precisions int8mm '
   f'--bench-list data/val_bench.txt --eval-list data/val_eval.txt --ann {ANN} '
   f'--out-dir results/int8_minmax --repeats {REPEATS} --iters {ITERS}')
""")

md("""
## Step 9c: GPU clock log (explains GPU-time differences between runs)
Logs SM clock, power and temperature while the FP16 engine runs with the baseline and the optimised CPU path.
""")
code("""
log = subprocess.Popen('exec nvidia-smi -i 0 --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu '
                       '--format=csv -lms 200 > results/gpu_clocks.csv', shell=True, cwd=REPO)
sh('./build/yolo_trt engines/yolov8s_fp16.engine bench data/val_bench.txt --iters 2000')
sh('./build/yolo_trt engines/yolov8s_fp16.engine bench data/val_bench.txt --iters 2000 --fast')
log.terminate(); log.wait()
sh('python scripts/plot_clocks.py results/gpu_clocks.csv results/gpu_clocks.png')
from IPython.display import Image, display
display(Image(filename=f'{REPO}/results/gpu_clocks.png'))
""")

md("""
## Step 10: Results
The table, correctness checks and headline numbers below are generated from this run's measurements only.
""")
code("""
from IPython.display import Markdown, Image, display
sh('python scripts/make_results.py results > /dev/null')
display(Markdown(open(f'{REPO}/results/results.md').read()))
display(Image(filename=f'{REPO}/results/latency.png'))
""")

md("## Step 11: Demo video / GIF (C++ app, FP16 engine)")
code("""
sh('python scripts/make_demo.py --bin build/yolo_trt --engine engines/yolov8s_fp16.engine '
   '--list data/calib.txt --out-dir results')
display(Image(filename=f'{REPO}/results/demo.gif'))
""")

md("## Step 12: Package everything for GitHub")
code("""
import zipfile
zpath = '/kaggle/working/yolov8-tensorrt-cpp-results.zip'
skip_dirs = {'build', 'engines', 'models', '__pycache__'}
with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for fn in files:
            if fn == 'demo_input.mp4':
                continue
            full = os.path.join(root, fn)
            z.write(full, os.path.join('yolov8-tensorrt-cpp', os.path.relpath(full, REPO)))
print('saved', zpath, round(os.path.getsize(zpath) / 1e6, 1), 'MB')
print(open(f'{REPO}/results/summary.json').read())
""")

md("""
## Next steps
1. Download `yolov8-tensorrt-cpp-results.zip` from the Output panel (or *Save Version → Save & Run All* to keep a public run).
2. Paste `results/results.md` into the README's *Results* section, add `results/latency.png` and `results/demo.gif`.
3. Push to GitHub and link this notebook from the README.
4. Fill the resume bullet with the numbers from **Headline** above, only those.
""")

nb = {'cells': cells, 'nbformat': 4, 'nbformat_minor': 5,
      'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                   'language_info': {'name': 'python'},
                   'kaggle': {'accelerator': 'nvidiaTeslaT4', 'isInternetEnabled': True, 'isGpuEnabled': True,
                              'language': 'python', 'sourceType': 'notebook'}}}
out = os.path.join(ROOT, 'notebooks', 'yolo_trt_kaggle.ipynb')
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, 'w') as f:
    json.dump(nb, f, indent=1)
print('wrote', out, len(cells), 'cells')
