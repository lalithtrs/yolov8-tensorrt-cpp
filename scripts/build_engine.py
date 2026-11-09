#!/usr/bin/env python3
"""Build a TensorRT engine (fp32 / fp16 / int8) from the ONNX model.

The INT8 calibrator keeps its input buffer on the GPU through PyTorch (no pycuda needed).
prep() must match the C++ preprocessing: letterbox to 640 (grey 114, centred), BGR->RGB, /255, CHW.
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch  # import before tensorrt so both share the CUDA libraries torch loaded
import tensorrt as trt

LOG = trt.Logger(trt.Logger.WARNING)


def prep(path, S=640):
    img = cv2.imread(path)
    h, w = img.shape[:2]
    sc = min(S / w, S / h)
    nw, nh = int(np.floor(w * sc + 0.5)), int(np.floor(h * sc + 0.5))  # same as C++ std::round
    px, py = (S - nw) // 2, (S - nh) // 2
    canvas = np.full((S, S, 3), 114, np.uint8)
    canvas[py:py + nh, px:px + nw] = cv2.resize(img, (nw, nh))
    x = canvas[:, :, ::-1].astype(np.float32) / 255.0
    return np.ascontiguousarray(x.transpose(2, 0, 1))[None]


def make_calibrator(kind, files, cache):
    base = {'entropy': trt.IInt8EntropyCalibrator2, 'minmax': trt.IInt8MinMaxCalibrator}[kind]

    class Calib(base):
        def __init__(self):
            base.__init__(self)
            self.files, self.i, self.cache = files, 0, cache
            self.buf = torch.empty((1, 3, 640, 640), dtype=torch.float32, device='cuda')

        def get_batch_size(self):
            return 1

        def get_batch(self, names):
            if self.i >= len(self.files):
                return None
            self.buf.copy_(torch.from_numpy(prep(self.files[self.i])))
            torch.cuda.synchronize()
            self.i += 1
            if self.i % 100 == 0:
                print(f'  calibration {self.i}/{len(self.files)}', flush=True)
            return [int(self.buf.data_ptr())]

        def read_calibration_cache(self):
            if os.path.exists(self.cache):
                with open(self.cache, 'rb') as f:
                    return f.read()
            return None

        def write_calibration_cache(self, c):
            with open(self.cache, 'wb') as f:
                f.write(c)

    return Calib()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('onnx')
    ap.add_argument('out')
    ap.add_argument('mode', choices=['fp32', 'fp16', 'int8'])
    ap.add_argument('--calib-list', help='image list for INT8 calibration')
    ap.add_argument('--calibrator', choices=['entropy', 'minmax'], default='entropy')
    ap.add_argument('--reuse-cache', action='store_true', help='reuse an existing calibration cache')
    ap.add_argument('--workspace-gb', type=float, default=2)
    ap.add_argument('--log', help='append build info to this JSON file')
    a = ap.parse_args()
    print('TensorRT', trt.__version__, '| mode', a.mode)

    b = trt.Builder(LOG)
    net = b.create_network(0)  # explicit batch is the default in TensorRT 10
    p = trt.OnnxParser(net, LOG)
    with open(a.onnx, 'rb') as f:
        if not p.parse(f.read()):
            for i in range(p.num_errors):
                print(p.get_error(i))
            sys.exit('ONNX parse failed')
    cfg = b.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(a.workspace_gb * (1 << 30)))
    if a.mode in ('fp16', 'int8'):
        cfg.set_flag(trt.BuilderFlag.FP16)  # lets layers that INT8 can't handle fall back to FP16
    cache = None
    if a.mode == 'int8':
        if not a.calib_list:
            sys.exit('--calib-list is required for int8')
        files = [l.strip() for l in open(a.calib_list) if l.strip()]
        cache = a.out + f'.{a.calibrator}.calib.cache'
        if os.path.exists(cache) and not a.reuse_cache:
            os.remove(cache)  # a stale cache silently reuses old scales
        cfg.set_flag(trt.BuilderFlag.INT8)
        cfg.int8_calibrator = make_calibrator(a.calibrator, files, cache)
        print(f'INT8 calibration: {a.calibrator}, {len(files)} images')

    t0 = time.time()
    plan = b.build_serialized_network(net, cfg)
    if plan is None:
        sys.exit('engine build failed (see TensorRT messages above)')
    with open(a.out, 'wb') as f:
        f.write(plan)
    secs = round(time.time() - t0, 1)
    mb = round(plan.nbytes / 1e6, 1)
    print(f'saved {a.out}: {mb} MB, built in {secs} s')

    if a.log:
        log = json.load(open(a.log)) if os.path.exists(a.log) else {}
        log[a.mode] = {'engine': a.out, 'size_mb': mb, 'build_s': secs, 'trt': trt.__version__,
                       'calibrator': a.calibrator if a.mode == 'int8' else None}
        json.dump(log, open(a.log, 'w'), indent=2)


if __name__ == '__main__':
    main()
