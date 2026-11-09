#!/usr/bin/env python3
"""Export YOLOv8 to ONNX: static 640x640, batch 1, opset 17, simplified."""
import argparse
import os
import shutil

import onnx
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='yolov8s.pt')
    ap.add_argument('--out', default='yolov8s.onnx')
    ap.add_argument('--imgsz', type=int, default=640)
    a = ap.parse_args()

    m = YOLO(a.weights)  # downloads the official weights on first use
    path = m.export(format='onnx', imgsz=a.imgsz, opset=17, simplify=True, dynamic=False, batch=1)
    if os.path.abspath(path) != os.path.abspath(a.out):
        shutil.copy(path, a.out)

    g = onnx.load(a.out).graph
    dims = lambda t: [d.dim_value or d.dim_param for d in t.type.tensor_type.shape.dim]
    for t in g.input:
        print('input ', t.name, dims(t))
    for t in g.output:
        print('output', t.name, dims(t))
    print('saved', a.out, round(os.path.getsize(a.out) / 1e6, 1), 'MB')


if __name__ == '__main__':
    main()
