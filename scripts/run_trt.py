#!/usr/bin/env python3
"""Run the C++ app for every engine: latency (baseline vs --fast CPU path) and COCO mAP.

Repeats are interleaved (repeat 1 of every variant, then repeat 2, ...) so slow drift in GPU clocks
affects all variants equally; the median of the repeats is reported.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coco_eval import coco_map, csv_to_json, image_id, read_list  # noqa: E402


def run(cmd, show=True):
    print('+', ' '.join(cmd), flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True)
    err = [l for l in p.stderr.splitlines() if l.strip()]
    if p.returncode != 0:
        print(p.stdout)
        print('\n'.join(err[-30:]))
        sys.exit(f'command failed with exit code {p.returncode}')
    if show:
        print('\n'.join(l for l in p.stdout.splitlines() if not l.startswith('JSON ')))
    return p.stdout


def bench(a, prec, fast):
    cmd = [a.bin, os.path.join(a.engines_dir, f'yolov8s_{prec}.engine'), 'bench', a.bench_list,
           '--iters', str(a.iters), '--warmup', str(a.warmup)] + (['--fast'] if fast else [])
    out = run(cmd)
    for line in out.splitlines():
        if line.startswith('JSON '):
            return json.loads(line[5:])
    sys.exit('no JSON line in bench output')


def load_csv(path):
    rows = []
    with open(path) as f:
        for line in f:
            v = line.strip().split(',')
            if len(v) == 7:
                rows.append((int(v[0]), int(v[1]), *map(float, v[2:])))
    return rows


def compare_paths(a, prec):
    """Check the --fast CPU path gives the same detections as the baseline path."""
    outs = {}
    for fast in (False, True):
        csv = os.path.join(a.det_dir, f'check_{prec}_{"fast" if fast else "base"}.csv')
        cmd = [a.bin, os.path.join(a.engines_dir, f'yolov8s_{prec}.engine'), 'coco', a.bench_list, csv]
        run(cmd + (['--fast'] if fast else []), show=False)
        outs[fast] = load_csv(csv)
        os.remove(csv)
    b, f = outs[False], outs[True]
    same_count = len(b) == len(f)
    max_box = max_score = 0.0
    if same_count:
        for rb, rf in zip(b, f):
            if rb[:2] != rf[:2]:
                same_count = False
                break
            max_box = max(max_box, *(abs(x - y) for x, y in zip(rb[2:6], rf[2:6])))
            max_score = max(max_score, abs(rb[6] - rf[6]))
    ok = same_count and max_box <= 0.05 and max_score <= 1e-4
    res = {'precision': prec, 'images': len(read_list(a.bench_list)), 'dets_baseline': len(b), 'dets_fast': len(f),
           'same_ids_and_classes': same_count, 'max_box_diff_px': max_box, 'max_score_diff': max_score, 'pass': ok}
    print('fast vs baseline:', res)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bin', required=True)
    ap.add_argument('--engines-dir', required=True)
    ap.add_argument('--precisions', nargs='+', default=['fp32', 'fp16', 'int8'])
    ap.add_argument('--bench-list', required=True)
    ap.add_argument('--eval-list', required=True)
    ap.add_argument('--ann', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--iters', type=int, default=500)
    ap.add_argument('--warmup', type=int, default=50)
    ap.add_argument('--skip-bench', action='store_true')
    ap.add_argument('--skip-coco', action='store_true')
    ap.add_argument('--det-dir', default='/tmp/dets', help='large detection files go here')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs(a.det_dir, exist_ok=True)

    if not a.skip_bench:
        runs = {}
        for r in range(a.repeats):
            print(f'\n===== benchmark repeat {r + 1}/{a.repeats} =====')
            for prec in a.precisions:
                for fast in (False, True):
                    runs.setdefault((prec, fast), []).append(bench(a, prec, fast))
        rows = []
        for (prec, fast), lst in runs.items():
            row = {'precision': prec, 'path': 'fast' if fast else 'baseline', 'repeats': len(lst)}
            for k in lst[0]:
                if k != 'path':
                    row[k] = round(statistics.median(x[k] for x in lst), 3)
            row['total_p50_spread'] = round(max(x['total_p50'] for x in lst) - min(x['total_p50'] for x in lst), 3)
            rows.append(row)
        check = compare_paths(a, 'fp16' if 'fp16' in a.precisions else a.precisions[0])
        with open(os.path.join(a.out_dir, 'trt_bench.json'), 'w') as f:
            json.dump({'rows': rows, 'fast_equals_baseline': check}, f, indent=2)
        print(json.dumps(rows, indent=2))

    if not a.skip_coco:
        ids = [image_id(p) for p in read_list(a.eval_list)]
        maps = {}
        for prec in a.precisions:
            csv = os.path.join(a.det_dir, f'det_trt_{prec}.csv')
            js = csv[:-4] + '.json'
            run([a.bin, os.path.join(a.engines_dir, f'yolov8s_{prec}.engine'), 'coco', a.eval_list, csv, '--fast'])
            n = csv_to_json(csv, js)
            os.remove(csv)
            print(f'TensorRT {prec}: {n} detections')
            maps[prec] = coco_map(js, a.ann, ids)
            print(f'TensorRT {prec} mAP:', maps[prec], flush=True)
        with open(os.path.join(a.out_dir, 'trt_map.json'), 'w') as f:
            json.dump(maps, f, indent=2)


if __name__ == '__main__':
    main()
