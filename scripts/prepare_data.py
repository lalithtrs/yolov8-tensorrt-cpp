#!/usr/bin/env python3
"""Find (or download) COCO val2017 and write the image lists.

Lists (sorted file names, disjoint so INT8 is never calibrated on images it is scored on):
  val_eval.txt   first N_EVAL images  -> mAP for every variant
  val_bench.txt  first N_BENCH images -> latency benchmarks
  calib.txt      last  N_CALIB images -> INT8 calibration
"""
import argparse
import glob
import json
import os
import subprocess
import sys

IMG_URL = 'http://images.cocodataset.org/zips/val2017.zip'
ANN_URL = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'


def walk(root, max_depth=6):
    root = os.path.abspath(root)
    base = root.count(os.sep)
    for d, dirs, files in os.walk(root):
        if d.count(os.sep) - base >= max_depth:
            dirs[:] = []
        yield d, dirs, files


def find_attached(root):
    """Look for a COCO 2017 dataset attached to the Kaggle notebook (Add Input)."""
    img_dir = ann = None
    if not os.path.isdir(root):
        return None, None
    for d, _, files in walk(root):
        if img_dir is None and os.path.basename(d) == 'val2017':
            if len(glob.glob(os.path.join(d, '*.jpg'))) >= 5000:
                img_dir = d
        if ann is None and 'instances_val2017.json' in files:
            ann = os.path.join(d, 'instances_val2017.json')
    return img_dir, ann


def sh(cmd):
    print('+', cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', required=True, help='where the lists and data.json go')
    ap.add_argument('--download-dir', default='/tmp/coco')
    ap.add_argument('--search', default='/kaggle/input')
    ap.add_argument('--n-eval', type=int, default=4500)
    ap.add_argument('--n-bench', type=int, default=300)
    ap.add_argument('--n-calib', type=int, default=500)
    a = ap.parse_args()

    img_dir, ann = find_attached(a.search)
    print('attached dataset:', img_dir or '-', ann or '-')
    dl = a.download_dir
    if img_dir is None:
        img_dir = os.path.join(dl, 'val2017')
        if len(glob.glob(os.path.join(img_dir, '*.jpg'))) < 5000:
            os.makedirs(dl, exist_ok=True)
            sh(f'cd {dl} && wget -q -c {IMG_URL} && unzip -q -o val2017.zip && rm -f val2017.zip')
    if ann is None:
        ann = os.path.join(dl, 'annotations', 'instances_val2017.json')
        if not os.path.exists(ann):
            os.makedirs(dl, exist_ok=True)
            sh(f'cd {dl} && wget -q -c {ANN_URL} && unzip -q -o annotations_trainval2017.zip '
               f'&& rm -f annotations_trainval2017.zip')

    files = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
    if len(files) != 5000:
        sys.exit(f'expected 5000 val2017 images in {img_dir}, found {len(files)}')
    if a.n_eval + a.n_calib > len(files):
        sys.exit('n_eval + n_calib must be <= 5000 so evaluation and calibration images stay disjoint')

    os.makedirs(a.out_dir, exist_ok=True)
    lists = {'val_eval.txt': files[:a.n_eval],
             'val_bench.txt': files[:a.n_bench],
             'calib.txt': files[-a.n_calib:]}
    for name, lst in lists.items():
        with open(os.path.join(a.out_dir, name), 'w') as f:
            f.write('\n'.join(lst) + '\n')
        print(f'{name}: {len(lst)} images')
    info = {'img_dir': img_dir, 'ann': ann, 'n_eval': a.n_eval, 'n_bench': a.n_bench, 'n_calib': a.n_calib}
    with open(os.path.join(a.out_dir, 'data.json'), 'w') as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
