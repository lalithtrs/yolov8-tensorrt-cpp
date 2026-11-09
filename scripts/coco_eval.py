#!/usr/bin/env python3
"""COCO bbox mAP helpers shared by the PyTorch baseline and the TensorRT runs.

CLI:  python coco_eval.py --ann instances_val2017.json --list data/val_eval.txt --csv det.csv --out det.json
"""
import argparse
import contextlib
import io
import json
import os

# YOLO class index (0..79) -> COCO category id (1..90, with gaps)
COCO91 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32,
          33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
          60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

_GT_CACHE = {}


def read_list(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def image_id(path):
    return int(os.path.splitext(os.path.basename(path))[0])


def csv_to_json(csv_path, out_path):
    """C++ 'coco' mode CSV (image_id,category_id,x,y,w,h,score) -> COCO results JSON."""
    res = []
    with open(csv_path) as f:
        for line in f:
            if not line.strip():
                continue
            i, c, x, y, w, h, s = line.strip().split(',')
            res.append({'image_id': int(i), 'category_id': int(c),
                        'bbox': [float(x), float(y), float(w), float(h)], 'score': float(s)})
    with open(out_path, 'w') as f:
        json.dump(res, f)
    return len(res)


def coco_map(det_json, ann, img_ids, verbose=True):
    """Return {'map50_95', 'map50'} (0..1) for the given image ids only."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with open(det_json) as f:
        if not json.load(f):
            print('WARNING: no detections in', det_json)
            return {'map50_95': 0.0, 'map50': 0.0}
    if ann not in _GT_CACHE:
        with contextlib.redirect_stdout(io.StringIO()):
            _GT_CACHE[ann] = COCO(ann)
    gt = _GT_CACHE[ann]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        dt = gt.loadRes(det_json)
        e = COCOeval(gt, dt, 'bbox')
        e.params.imgIds = sorted(img_ids)
        e.evaluate()
        e.accumulate()
        e.summarize()
    if verbose:
        print('\n'.join(l for l in buf.getvalue().splitlines() if l.strip().startswith('Average Precision')))
    return {'map50_95': float(e.stats[0]), 'map50': float(e.stats[1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ann', required=True)
    ap.add_argument('--list', required=True, help='image list that was evaluated')
    ap.add_argument('--csv', help='C++ CSV to convert first')
    ap.add_argument('--out', required=True, help='COCO results JSON (written if --csv is given)')
    a = ap.parse_args()
    if a.csv:
        csv_to_json(a.csv, a.out)
    ids = [image_id(p) for p in read_list(a.list)]
    print(json.dumps(coco_map(a.out, a.ann, ids)))


if __name__ == '__main__':
    main()
