#!/usr/bin/env python3
"""Combine all measurements into results/results.md, results.csv, summary.json and latency.png."""
import csv
import json
import os
import sys

R = sys.argv[1] if len(sys.argv) > 1 else 'results'


def load(name):
    p = os.path.join(R, name)
    return json.load(open(p)) if os.path.exists(p) else None


def f(v, nd=2):
    return '-' if v is None else f'{v:.{nd}f}'


def main():
    env, pt, tb, tm, eng = (load(n) for n in ('env.json', 'pytorch.json', 'trt_bench.json', 'trt_map.json',
                                              'engines.json'))
    rows = []  # (label, pre, gpu/infer, post, total50, total95, fps, map50_95, map50)
    for k, lab in (('fp32', 'PyTorch FP32 (Ultralytics)'), ('fp16', 'PyTorch FP16 (Ultralytics)')):
        if pt and k in pt:
            s = pt[k]
            rows.append([lab, s['pre_p50'], s['infer_p50'], s['post_p50'], s['total_p50'], s['total_p95'], s['fps'],
                         s.get('map50_95'), s.get('map50')])
    bench = {(r['precision'], r['path']): r for r in tb['rows']} if tb else {}
    for prec in ('fp32', 'fp16', 'int8'):
        for path, suffix in (('baseline', 'C++'), ('fast', 'C++, optimised CPU')):
            b = bench.get((prec, path))
            if not b:
                continue
            mp = (tm or {}).get(prec, {})
            rows.append([f"TensorRT {prec.upper()}{' entropy calib' if prec == 'int8' else ''} ({suffix})", b['pre_p50'], b['gpu_p50'], b['post_p50'],
                         b['total_p50'], b['total_p95'], b['fps'], mp.get('map50_95'), mp.get('map50')])

    mm_b, mm_m = load('int8_minmax/trt_bench.json'), load('int8_minmax/trt_map.json')
    if mm_b and mm_m:  # optional Step 9b experiment
        mb = {r['path']: r for r in mm_b['rows']}
        mp = mm_m.get('int8mm', {})
        for path, suffix in (('baseline', 'C++'), ('fast', 'C++, optimised CPU')):
            if path in mb:
                b = mb[path]
                rows.append([f'TensorRT INT8 MinMax calib ({suffix})', b['pre_p50'], b['gpu_p50'], b['post_p50'],
                             b['total_p50'], b['total_p95'], b['fps'], mp.get('map50_95'), mp.get('map50')])

    hdr = ['Variant', 'Pre ms', 'GPU ms', 'Post ms', 'Total P50 ms', 'Total P95 ms', 'FPS', 'mAP50-95', 'mAP50']
    md = ['# Results', '']
    if env:
        md += [f"Measured on **{env.get('gpu', '?')}**, CPU {env.get('cpu', '?')} ({env.get('cpu_cores')} cores). "
               f"TensorRT {env.get('tensorrt', '?')}, PyTorch {env.get('torch', '?')} (CUDA {env.get('torch_cuda', '?')}), "
               f"Ultralytics {env.get('ultralytics', '?')}, OpenCV (C++) {env.get('opencv_cpp') or '?'}.", '']
    md += ['All latencies are per frame at batch 1, 640x640, median of repeated runs (P50 / P95 over 500 timed '
           'frames after 50 warm-up frames). "GPU ms" for TensorRT includes the host-to-device and device-to-host '
           'copies; for PyTorch it is Ultralytics\' model inference time. mAP is COCO val2017 '
           f"({(load('../data/data.json') or load('data.json') or {}).get('n_eval', 'N')} images, not used for INT8 calibration).", '',
           '| ' + ' | '.join(hdr) + ' |', '|' + '---|' * len(hdr)]
    for r in rows:
        md.append('| ' + ' | '.join([r[0]] + [f(v) for v in r[1:6]] + [f(r[6], 1)] +
                                    [f(None if v is None else 100 * v, 1) for v in r[7:]]) + ' |')

    if tb:
        md += ['', '## TensorRT GPU stage breakdown (P50 ms, optimised CPU path)', '',
               '| Precision | H2D copy | Inference | D2H copy | GPU total |', '|---|---|---|---|---|']
        for prec in ('fp32', 'fp16', 'int8'):
            b = bench.get((prec, 'fast'))
            if b:
                md.append(f"| {prec.upper()} | {f(b['h2d_p50'])} | {f(b['infer_p50'])} | {f(b['d2h_p50'])} | "
                          f"{f(b['gpu_p50'])} |")

    # Checks
    summ = {}
    md += ['', '## Correctness checks', '']

    def chk(ok, text):
        md.append(f"- **{'PASS' if ok else 'CHECK'}**: {text}")

    ptm = (pt or {}).get('fp32', {}).get('map50_95')
    t32, t16, t8 = ((tm or {}).get(p, {}).get('map50_95') for p in ('fp32', 'fp16', 'int8'))
    if ptm is not None and t32 is not None:
        d = 100 * (t32 - ptm)
        summ['trt_fp32_vs_pt_map_diff'] = round(d, 2)
        chk(abs(d) <= 0.5, f'TensorRT FP32 mAP vs PyTorch FP32: {d:+.2f} points (expected within 0.5)')
    if t32 is not None and t16 is not None:
        d = 100 * (t16 - t32)
        summ['fp16_vs_fp32_map_diff'] = round(d, 2)
        chk(abs(d) <= 0.3, f'TensorRT FP16 vs FP32 mAP: {d:+.2f} points (expected almost identical)')
    if t32 is not None and t8 is not None:
        d = 100 * (t8 - t32)
        summ['int8_vs_fp32_map_diff'] = round(d, 2)
        chk(d > -2.0, f'TensorRT INT8 (entropy calibration) vs FP32 mAP: {d:+.2f} points (a drop is expected; report it as measured)')
    if mm_m and t32 is not None and mm_m.get('int8mm', {}).get('map50_95') is not None:
        d = 100 * (mm_m['int8mm']['map50_95'] - t32)
        summ['int8_minmax_vs_fp32_map_diff'] = round(d, 2)
        chk(d > -2.0, f'TensorRT INT8 (MinMax calibration) vs FP32 mAP: {d:+.2f} points')
    if tb:
        c = tb['fast_equals_baseline']
        chk(c['pass'], f"Optimised CPU path gives the same detections as the baseline path on {c['images']} images "
                       f"({c['dets_fast']} vs {c['dets_baseline']} boxes, max box diff {c['max_box_diff_px']:.3f} px)")

    # Headline numbers
    md += ['', '## Headline', '']
    ptt = (pt or {}).get('fp32', {}).get('total_p50')
    for prec in ('fp32', 'fp16', 'int8'):
        b, base = bench.get((prec, 'fast')), bench.get((prec, 'baseline'))
        if b and ptt:
            sp = ptt / b['total_p50']
            summ[f'speedup_{prec}_vs_pt_fp32'] = round(sp, 2)
            md.append(f"- TensorRT {prec.upper()} (C++, optimised): {b['total_p50']:.2f} ms/frame vs PyTorch FP32 "
                      f"{ptt:.2f} ms -> **{sp:.1f}x** end-to-end")
        if b and base:
            md.append(f"  - CPU-path optimisation at {prec.upper()}: pre {base['pre_p50']:.2f} -> {b['pre_p50']:.2f} ms, "
                      f"post {base['post_p50']:.2f} -> {b['post_p50']:.2f} ms, total {base['total_p50']:.2f} -> "
                      f"{b['total_p50']:.2f} ms")
    for prec in ('fp16', 'int8'):
        bb, bf = bench.get((prec, 'baseline')), bench.get((prec, 'fast'))
        if bb:
            cpu = bb['pre_p50'] + bb['post_p50']
            summ[f'{prec}_baseline_cpu_share'] = round(cpu / bb['total_p50'], 3)
            md.append(f"- Bottleneck before the CPU optimisation at {prec.upper()}: CPU pre + post = {cpu:.2f} ms "
                      f"({100 * cpu / bb['total_p50']:.0f}% of {bb['total_p50']:.2f} ms) vs GPU {bb['gpu_p50']:.2f} ms")
        if bf:
            copies = bf['h2d_p50'] + bf['d2h_p50']
            summ[f'{prec}_fast_copy_share'] = round(copies / bf['total_p50'], 3)
            md.append(f"- After it, at {prec.upper()}: inference {bf['infer_p50']:.2f} ms, host-device copies "
                      f"{copies:.2f} ms ({100 * copies / bf['total_p50']:.0f}%), CPU {bf['pre_p50'] + bf['post_p50']:.2f} ms "
                      f"of {bf['total_p50']:.2f} ms")
    if eng:
        md += ['', '## Engines', '', '| Precision | Size MB | Build s |', '|---|---|---|']
        for k, v in eng.items():
            cal = f" ({v['calibrator']} calibration)" if v.get('calibrator') else ''
            md.append(f"| {k.upper()}{cal} | {v['size_mb']} | {v['build_s']} |")
        mm_e = load('int8_minmax_engine.json')
        if mm_e and 'int8' in mm_e:
            v = mm_e['int8']
            md.append(f"| INT8 ({v.get('calibrator', 'minmax')} calibration) | {v['size_mb']} | {v['build_s']} |")

    with open(os.path.join(R, 'results.md'), 'w') as fh:
        fh.write('\n'.join(md) + '\n')
    with open(os.path.join(R, 'results.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(hdr)
        w.writerows(rows)
    with open(os.path.join(R, 'summary.json'), 'w') as fh:
        json.dump(summ, fh, indent=2)
    plot(rows, os.path.join(R, 'latency.png'))
    print('\n'.join(md))


def plot(rows, out):
    """Horizontal stacked bars: where each frame's time goes (P50 per stage)."""
    if not rows:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    INK, MUTED, GRID, SURFACE = '#1f1f1e', '#6b6a63', '#e4e3dd', '#ffffff'
    SERIES = [('Preprocess (CPU)', '#2a78d6'), ('Inference / GPU', '#eb6834'), ('Postprocess (CPU)', '#1baf7a'),
              ('Other (framework overhead, not in a stage)', '#b9b8b0')]
    labels = [r[0] for r in rows][::-1]
    stages = [[r[i] for r in rows][::-1] for i in (1, 2, 3)]
    totals = [r[4] for r in rows][::-1]
    stages.append([max(0.0, t - a - b - c) for t, a, b, c in zip(totals, *stages)])

    fig, ax = plt.subplots(figsize=(10, 0.55 * len(rows) + 1.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    left = [0.0] * len(rows)
    for (name, color), vals in zip(SERIES, stages):
        ax.barh(labels, vals, left=left, height=0.55, color=color, edgecolor=SURFACE, linewidth=2, label=name)
        left = [l + v for l, v in zip(left, vals)]
    for y, (stack, tot) in enumerate(zip(left, totals)):
        ax.text(stack + max(left) * 0.01, y, f'{tot:.1f} ms total', va='center', fontsize=8, color=MUTED)
    ax.set_xlabel('Median latency per frame (ms)', color=MUTED, fontsize=9)
    ax.set_xlim(0, max(left) * 1.18)
    ax.tick_params(colors=INK, labelsize=8, length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title('YOLOv8s per-frame latency by stage (batch 1, 640x640)', loc='left', fontsize=11, color=INK)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc='lower left', ncol=4, frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out, facecolor=SURFACE, bbox_inches='tight')
    print('saved', out)


if __name__ == '__main__':
    main()
