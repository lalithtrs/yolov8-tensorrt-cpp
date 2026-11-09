#!/usr/bin/env python3
"""Run the whole experiment on a local machine, the same steps as the Kaggle notebook.

Everything (data lists, models, engines, results) goes to a run folder (default runs/local/), so the
reference results in results/ are never overwritten. Compare afterwards with:
    python scripts/compare_results.py results runs/local/results

Examples:
    python scripts/run_all.py --quick                     # ~20 min smoke run
    python scripts/run_all.py                             # full run (~1-1.5 h)
    python scripts/run_all.py --steps coco,results        # re-run only some steps
    python scripts/run_all.py --coco-dir /data/coco       # COCO already downloaded (val2017/ + annotations/)
"""
import argparse
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEPS = ['check', 'build', 'data', 'onnx', 'engines', 'pytorch', 'bench', 'coco', 'minmax', 'clocks', 'results',
         'demo']
HELP = {'check': 'GPU, driver and package versions', 'build': 'build the C++ app + TensorRT smoke test',
        'data': 'COCO val2017 + image lists', 'onnx': 'export YOLOv8s to ONNX',
        'engines': 'TensorRT FP32/FP16/INT8 engines', 'pytorch': 'PyTorch baseline (latency + mAP)',
        'bench': 'TensorRT C++ latency', 'coco': 'TensorRT C++ mAP',
        'minmax': 'INT8 MinMax calibration experiment', 'clocks': 'GPU clock/power log',
        'results': 'tables, checks, charts', 'demo': 'demo video + GIF'}


def sh(cmd, cwd=REPO):
    print(f'$ {cmd}', flush=True)
    r = subprocess.run(cmd, shell=True, cwd=cwd)
    if r.returncode != 0:
        sys.exit(f'\nFAILED (exit {r.returncode}): {cmd}\nFix the error above, then resume with --steps <this step>,...')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', default='runs/local', help='output folder (relative to the repo or absolute)')
    ap.add_argument('--coco-dir', default=os.path.expanduser('~/datasets/coco'),
                    help='where COCO val2017 is (or will be downloaded, ~1 GB)')
    ap.add_argument('--quick', action='store_true', help='500 eval images, 1 benchmark repeat')
    ap.add_argument('--steps', help='comma-separated subset of: ' + ','.join(STEPS))
    ap.add_argument('--skip', default='', help='comma-separated steps to skip')
    ap.add_argument('--rebuild-engines', action='store_true', help='rebuild engines even if they exist')
    ap.add_argument('--gpu', default=os.environ.get('CUDA_VISIBLE_DEVICES', '0'), help='GPU index to use')
    a = ap.parse_args()

    steps = a.steps.split(',') if a.steps else list(STEPS)
    steps = [s for s in steps if s not in a.skip.split(',')]
    bad = [s for s in steps if s not in STEPS]
    if bad:
        sys.exit(f'unknown steps {bad}; choose from {STEPS}')

    run = a.run_dir if os.path.isabs(a.run_dir) else os.path.join(REPO, a.run_dir)
    D, M, E, R = (os.path.join(run, d) for d in ('data', 'models', 'engines', 'results'))
    for d in (D, M, E, R):
        os.makedirs(d, exist_ok=True)
    n_eval, n_calib = (500, 200) if a.quick else (4500, 500)
    repeats, iters = (1, 200) if a.quick else (3, 500)
    os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu  # same single GPU for every step (and nvidia-smi -i below)
    py = sys.executable
    ann = lambda: json.load(open(os.path.join(D, 'data.json')))['ann']
    common = f'--bench-list {D}/val_bench.txt --eval-list {D}/val_eval.txt'

    print(f'run folder: {run}\nsteps: {", ".join(steps)}\nmode: {"QUICK" if a.quick else "full"}\n')
    times = {}
    for step in steps:
        t0 = time.time()
        print(f'\n{"=" * 70}\n== {step}: {HELP[step]}\n{"=" * 70}', flush=True)
        if step == 'check':
            sh(f'nvidia-smi -i {a.gpu}')
            sh(f'{py} -c "import torch, tensorrt; assert torch.cuda.is_available(), \'no CUDA GPU visible\'; '
               f'print(\'torch\', torch.__version__, \'| tensorrt\', tensorrt.__version__, \'| GPU\', '
               f'torch.cuda.get_device_name(0))"')
            sh(f'{py} scripts/collect_env.py {R}/env.json')
        elif step == 'build':
            sh(f'{py} scripts/setup_cpp.py')
            sh('./build/hello_trt')
        elif step == 'data':
            sh(f'{py} scripts/prepare_data.py --out-dir {D} --download-dir {a.coco_dir} --search {a.coco_dir} '
               f'--n-eval {n_eval} --n-calib {n_calib}')
        elif step == 'onnx':
            sh(f'{py} {REPO}/scripts/export_onnx.py --weights yolov8s.pt --out yolov8s.onnx', cwd=M)
        elif step == 'engines':
            for prec in ('fp32', 'fp16', 'int8'):
                eng = f'{E}/yolov8s_{prec}.engine'
                if os.path.exists(eng) and not a.rebuild_engines:
                    print(f'{eng} exists, skipping (use --rebuild-engines to rebuild)')
                    continue
                extra = f' --calib-list {D}/calib.txt' if prec == 'int8' else ''
                sh(f'{py} scripts/build_engine.py {M}/yolov8s.onnx {eng} {prec}{extra} --log {R}/engines.json')
        elif step == 'pytorch':
            sh(f'{py} scripts/pytorch_baseline.py --weights {M}/yolov8s.pt {common} --ann {ann()} '
               f'--out-dir {R} --iters {iters}')
        elif step == 'bench':
            sh(f'{py} scripts/run_trt.py --bin build/yolo_trt --engines-dir {E} {common} --ann {ann()} '
               f'--out-dir {R} --repeats {repeats} --iters {iters} --skip-coco')
        elif step == 'coco':
            sh(f'{py} scripts/run_trt.py --bin build/yolo_trt --engines-dir {E} {common} --ann {ann()} '
               f'--out-dir {R} --skip-bench')
        elif step == 'minmax':
            eng = f'{E}/yolov8s_int8mm.engine'
            if not os.path.exists(eng) or a.rebuild_engines:
                sh(f'{py} scripts/build_engine.py {M}/yolov8s.onnx {eng} int8 --calib-list {D}/calib.txt '
                   f'--calibrator minmax --log {R}/int8_minmax_engine.json')
            sh(f'{py} scripts/run_trt.py --bin build/yolo_trt --engines-dir {E} --precisions int8mm {common} '
               f'--ann {ann()} --out-dir {R}/int8_minmax --repeats {repeats} --iters {iters}')
        elif step == 'clocks':
            log = subprocess.Popen(f'exec nvidia-smi -i {a.gpu} --query-gpu=timestamp,clocks.sm,power.draw,'
                                   f'temperature.gpu --format=csv -lms 200 > {R}/gpu_clocks.csv', shell=True)
            try:
                for extra in ('', ' --fast'):
                    sh(f'./build/yolo_trt {E}/yolov8s_fp16.engine bench {D}/val_bench.txt --iters 2000{extra}')
            finally:
                log.terminate()
                log.wait()
            sh(f'{py} scripts/plot_clocks.py {R}/gpu_clocks.csv {R}/gpu_clocks.png')
        elif step == 'results':
            sh(f'{py} scripts/make_results.py {R}')
        elif step == 'demo':
            sh(f'{py} scripts/make_demo.py --bin build/yolo_trt --engine {E}/yolov8s_fp16.engine '
               f'--list {D}/calib.txt --out-dir {R}')
        times[step] = time.time() - t0

    print(f'\n{"=" * 70}\nDone. Step times:')
    for s, t in times.items():
        print(f'  {s:8s} {t / 60:6.1f} min')
    print(f'\nResults: {R}/results.md\nCompare with the reference run:\n'
          f'  python scripts/compare_results.py results {os.path.relpath(R, REPO)}')


if __name__ == '__main__':
    main()
