#!/usr/bin/env python3
"""Plot the T4 SM clock during the two FP16 benchmark runs (baseline vs --fast CPU path).

Input: results/gpu_clocks.csv from
  nvidia-smi -i 0 --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu --format=csv -lms 200
Usage: python3 scripts/plot_clocks.py results/gpu_clocks.csv results/gpu_clocks.png
The two runs are found as the two stretches of high power draw (> 55 W) in the log.
"""
import csv
import statistics as st
import sys
from datetime import datetime


def load(path):
    rows = []
    for r in csv.reader(open(path)):
        if not r or r[0].startswith('timestamp'):
            continue
        t = datetime.strptime(r[0].strip(), '%Y/%m/%d %H:%M:%S.%f')
        rows.append((t, float(r[1].split()[0]), float(r[2].split()[0]), float(r[3])))
    t0 = rows[0][0]
    return [((t - t0).total_seconds(), c, p, T) for t, c, p, T in rows]


def runs(rows, thr=55.0, min_len=10):
    """Stretches of consecutive samples with power above thr (allowing 1-sample dips)."""
    out, start = [], None
    for i, r in enumerate(rows):
        hot = r[2] > thr or (start is not None and i + 1 < len(rows) and rows[i + 1][2] > thr)
        if hot and start is None:
            start = i
        if not hot and start is not None:
            if i - start >= min_len:
                out.append((start, i))
            start = None
    if start is not None and len(rows) - start >= min_len:
        out.append((start, len(rows)))
    return out


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else 'results/gpu_clocks.csv'
    out = sys.argv[2] if len(sys.argv) > 2 else 'results/gpu_clocks.png'
    rows = load(src)
    found = runs(rows)
    names = ['baseline CPU path', 'optimised CPU path (--fast)']
    stats = []
    for (a, b), n in zip(found, names):
        seg = rows[a:b]
        stats.append((n, seg[0][0], seg[-1][0], st.median(r[1] for r in seg), st.median(r[2] for r in seg),
                      seg[0][3], seg[-1][3]))
        print(f'{n}: {seg[-1][0] - seg[0][0]:.1f} s, clock median {stats[-1][3]:.0f} MHz, '
              f'power median {stats[-1][4]:.1f} W, temp {seg[0][3]:.0f}->{seg[-1][3]:.0f} C')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    INK, MUTED, GRID, SURFACE, LINE, BAND = '#1f1f1e', '#6b6a63', '#e4e3dd', '#ffffff', '#2a78d6', '#f1f0eb'
    fig, ax = plt.subplots(figsize=(10, 3.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for n, t0, t1, clk, pw, ta, tb in stats:
        ax.axvspan(t0, t1, color=BAND, zorder=0)
        ax.text((t0 + t1) / 2, 1660, f'{n}\nmedian {clk:.0f} MHz, {pw:.0f} W, {ta:.0f}->{tb:.0f} C',
                ha='center', va='bottom', fontsize=8, color=INK)
    ax.plot([r[0] for r in rows], [r[1] for r in rows], color=LINE, linewidth=2)
    ax.axhline(1590, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(rows[0][0], 1600, 'T4 max boost 1590 MHz', fontsize=8, color=MUTED, va='bottom')
    ax.set_ylim(0, 1900)
    ax.set_ylabel('SM clock (MHz)', color=MUTED, fontsize=9)
    ax.set_xlabel('Time (s)', color=MUTED, fontsize=9)
    ax.tick_params(colors=INK, labelsize=8, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title('T4 clock during FP16 benchmarks: same GPU work, lower clock when the GPU is busier',
                 loc='left', fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    print('saved', out)


if __name__ == '__main__':
    main()
