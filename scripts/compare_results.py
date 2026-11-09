#!/usr/bin/env python3
"""Compare your run against the reference results.

    python scripts/compare_results.py results runs/local/results [--out runs/local/results/comparison.md]

Accuracy (mAP) should reproduce on any GPU. Absolute latency only reproduces on the same GPU type, so for a
different GPU compare the speedups (each variant vs your own PyTorch FP32), not the milliseconds.
"""
import argparse
import json
import os

# Tolerances for "reproduced" (mAP points). INT8 is looser: calibration and kernel choice can vary.
TOL = {'PyTorch FP32': 0.5, 'PyTorch FP16': 0.5, 'TensorRT FP32': 0.5, 'TensorRT FP16': 0.5,
       'TensorRT INT8 entropy': 1.0, 'TensorRT INT8 MinMax': 1.0}


def load(d, name):
    p = os.path.join(d, name)
    return json.load(open(p)) if os.path.exists(p) else None


def collect(d):
    pt, tm, tb = load(d, 'pytorch.json') or {}, load(d, 'trt_map.json') or {}, load(d, 'trt_bench.json')
    mm_m, mm_b = load(d, 'int8_minmax/trt_map.json') or {}, load(d, 'int8_minmax/trt_bench.json')
    bench = {(r['precision'], r['path']): r for r in (tb or {}).get('rows', [])}
    bench.update({('int8mm', r['path']): r for r in (mm_b or {}).get('rows', [])})
    m = {'PyTorch FP32': pt.get('fp32', {}).get('map50_95'), 'PyTorch FP16': pt.get('fp16', {}).get('map50_95'),
         'TensorRT FP32': tm.get('fp32', {}).get('map50_95'), 'TensorRT FP16': tm.get('fp16', {}).get('map50_95'),
         'TensorRT INT8 entropy': tm.get('int8', {}).get('map50_95'),
         'TensorRT INT8 MinMax': mm_m.get('int8mm', {}).get('map50_95')}
    lat = {'PyTorch FP32': pt.get('fp32', {}).get('total_p50'), 'PyTorch FP16': pt.get('fp16', {}).get('total_p50')}
    for key, lab in (('fp32', 'FP32'), ('fp16', 'FP16'), ('int8', 'INT8 entropy'), ('int8mm', 'INT8 MinMax')):
        for path, suf in (('baseline', ''), ('fast', ', optimised CPU')):
            r = bench.get((key, path))
            lat[f'TensorRT {lab}{suf}'] = r['total_p50'] if r else None
    checks = [x['fast_equals_baseline']['pass'] for x in (tb, mm_b) if x and 'fast_equals_baseline' in x]
    return m, lat, checks, load(d, 'env.json') or {}


def fmt(v, nd=2):
    return '-' if v is None else f'{v:.{nd}f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('reference')
    ap.add_argument('yours')
    ap.add_argument('--out', help='also write the comparison as Markdown')
    a = ap.parse_args()
    rm, rl, _, renv = collect(a.reference)
    ym, yl, ychecks, yenv = collect(a.yours)
    same_gpu = renv.get('gpu', '').split(',')[0] == yenv.get('gpu', '').split(',')[0]
    md = ['# Reproduction check', '',
          f"- Reference: {renv.get('gpu', '?')}, TensorRT {renv.get('tensorrt', '?')}, torch {renv.get('torch', '?')}",
          f"- Yours:     {yenv.get('gpu', '?')}, TensorRT {yenv.get('tensorrt', '?')}, torch {yenv.get('torch', '?')}",
          '', '## Accuracy (mAP50-95, COCO val2017)', '',
          '| Variant | Reference | Yours | Diff | Result |', '|---|---|---|---|---|']
    acc_ok = []
    for k, tol in TOL.items():
        r, y = rm.get(k), ym.get(k)
        if r is None or y is None:
            md.append(f'| {k} | {fmt(r and 100 * r)} | {fmt(y and 100 * y)} | - | not run |')
            continue
        d = 100 * (y - r)
        ok = abs(d) <= tol
        acc_ok.append(ok)
        md.append(f"| {k} | {100 * r:.2f} | {100 * y:.2f} | {d:+.2f} | {'reproduced' if ok else f'differs (> {tol})'} |")

    md += ['', '## Latency (total per frame, P50 ms) and speedup vs PyTorch FP32', '',
           'Same GPU type: ms should be within about 10%. Different GPU: compare the speedup columns.' if same_gpu
           else f"**Different GPU** ({yenv.get('gpu', '?').split(',')[0]} vs {renv.get('gpu', '?').split(',')[0]}): "
                'absolute ms are not comparable; compare the speedup columns.', '',
           '| Variant | Reference ms | Yours ms | Reference speedup | Your speedup |', '|---|---|---|---|---|']
    rb, yb = rl.get('PyTorch FP32'), yl.get('PyTorch FP32')
    for k in rl:
        r, y = rl.get(k), yl.get(k)
        rs = f'{rb / r:.1f}x' if r and rb else '-'
        ys = f'{yb / y:.1f}x' if y and yb else '-'
        md.append(f'| {k} | {fmt(r)} | {fmt(y)} | {rs} | {ys} |')

    md += ['', '## Summary', '']
    if acc_ok:
        md.append(f"- Accuracy: {sum(acc_ok)}/{len(acc_ok)} variants within tolerance "
                  f"({'REPRODUCED' if all(acc_ok) else 'see differences above'})")
    if ychecks:
        md.append(f"- Optimised CPU path gives the same detections as the baseline: "
                  f"{'PASS' if all(ychecks) else 'FAIL'}")
    md.append(f"- GPU: {'same type as the reference' if same_gpu else 'different from the reference'}")
    text = '\n'.join(md)
    print(text)
    if a.out:
        open(a.out, 'w').write(text + '\n')
        print('\nsaved', a.out)


if __name__ == '__main__':
    main()
