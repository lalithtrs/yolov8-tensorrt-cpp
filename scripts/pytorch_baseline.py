#!/usr/bin/env python3
"""PyTorch (Ultralytics) baseline: per-stage latency and COCO mAP, FP32 and FP16.

Note: Ultralytics pads to a multiple of 32 ("rect" letterbox) instead of a full 640x640 square and
runs NMS on the GPU, so its stages are not identical to the C++ pipeline. That is stated in the README.
"""
import argparse
import json
import logging
import os
import sys
import time
import warnings

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils import LOGGER

# Newer Ultralytics prints a deprecation warning for `half` on EVERY predict call; it still works.
# Silence it so progress lines stay visible and the notebook output doesn't flood.
LOGGER.setLevel(logging.ERROR)
warnings.filterwarnings('ignore', message=".*half.*")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coco_eval import COCO91, coco_map, image_id, read_list  # noqa: E402


def p(a, q):
    return round(float(np.percentile(a, q)), 3)


def speed(m, imgs, half, warmup, iters):
    kw = dict(imgsz=640, conf=0.25, iou=0.45, device=0, verbose=False, **({'half': True} if half else {}))
    for i in range(warmup):
        m.predict(imgs[i % len(imgs)], **kw)
    pre, inf, post, tot = [], [], [], []
    for i in range(iters):
        torch.cuda.synchronize()
        t = time.perf_counter()
        r = m.predict(imgs[i % len(imgs)], **kw)[0]
        torch.cuda.synchronize()
        tot.append((time.perf_counter() - t) * 1000)
        pre.append(r.speed['preprocess'])
        inf.append(r.speed['inference'])
        post.append(r.speed['postprocess'])
    res = {'pre_p50': p(pre, 50), 'infer_p50': p(inf, 50), 'infer_p95': p(inf, 95), 'post_p50': p(post, 50),
           'total_p50': p(tot, 50), 'total_p95': p(tot, 95)}
    res['fps'] = round(1000 / res['total_p50'], 1)
    return res


def accuracy(m, files, half, out_json, ann):
    res = []
    for n, f in enumerate(files, 1):
        r = m.predict(f, imgsz=640, conf=0.001, iou=0.7, max_det=300, device=0, verbose=False,
                      **({'half': True} if half else {}))[0]
        iid = image_id(f)
        b = r.boxes
        for (x1, y1, x2, y2), s, c in zip(b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy()):
            res.append({'image_id': iid, 'category_id': COCO91[int(c)],
                        'bbox': [round(float(x1), 2), round(float(y1), 2), round(float(x2 - x1), 2),
                                 round(float(y2 - y1), 2)], 'score': round(float(s), 5)})
        if n % 1000 == 0:
            print(f'  {n}/{len(files)} images', flush=True)
    with open(out_json, 'w') as fh:
        json.dump(res, fh)
    return coco_map(out_json, ann, [image_id(f) for f in files])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--bench-list', required=True)
    ap.add_argument('--eval-list', required=True)
    ap.add_argument('--ann', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--iters', type=int, default=500)
    ap.add_argument('--warmup', type=int, default=50)
    ap.add_argument('--skip-map', action='store_true')
    ap.add_argument('--det-dir', default='/tmp/dets', help='large detection JSONs go here')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs(a.det_dir, exist_ok=True)

    m = YOLO(a.weights)
    imgs = [cv2.imread(f) for f in read_list(a.bench_list)]
    out = {}
    for half in (False, True):
        name = 'fp16' if half else 'fp32'
        s = speed(m, imgs, half, a.warmup, a.iters)
        print(f'PyTorch {name} speed:', s, flush=True)
        if not a.skip_map:
            mp = accuracy(m, read_list(a.eval_list), half, os.path.join(a.det_dir, f'det_pt_{name}.json'), a.ann)
            print(f'PyTorch {name} mAP:', mp, flush=True)
            s.update(mp)
        out[name] = s
    with open(os.path.join(a.out_dir, 'pytorch.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
