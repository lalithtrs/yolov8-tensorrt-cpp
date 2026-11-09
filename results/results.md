# Results

Measured on **Tesla T4, 580.178.04, 15360 MiB**, CPU Intel(R) Xeon(R) CPU @ 2.00GHz (4 cores). TensorRT 10.16.1.11, PyTorch 2.11.0+cu130 (CUDA 13.0), Ultralytics 8.4.208, OpenCV (C++) 4.6.0.

All latencies are per frame at batch 1, 640x640, median of repeated runs (P50 / P95 over 500 timed frames after 50 warm-up frames). "GPU ms" for TensorRT includes the host-to-device and device-to-host copies; for PyTorch it is Ultralytics' model inference time. mAP is COCO val2017 (4500 images, not used for INT8 calibration).

| Variant | Pre ms | GPU ms | Post ms | Total P50 ms | Total P95 ms | FPS | mAP50-95 | mAP50 |
|---|---|---|---|---|---|---|---|---|
| PyTorch FP32 (Ultralytics) | 0.81 | 11.83 | 1.16 | 14.48 | 18.91 | 69.0 | 44.4 | 60.7 |
| PyTorch FP16 (Ultralytics) | 0.84 | 8.11 | 1.22 | 10.90 | 12.64 | 91.7 | 44.3 | 60.7 |
| TensorRT FP32 (C++) | 2.33 | 7.98 | 1.29 | 11.67 | 12.21 | 85.7 | 44.5 | 61.0 |
| TensorRT FP32 (C++, optimised CPU) | 0.80 | 8.81 | 0.29 | 9.99 | 11.02 | 100.1 | 44.5 | 61.0 |
| TensorRT FP16 (C++) | 2.34 | 2.92 | 1.30 | 6.61 | 7.19 | 151.3 | 44.5 | 60.9 |
| TensorRT FP16 (C++, optimised CPU) | 0.78 | 3.16 | 0.28 | 4.25 | 5.21 | 235.1 | 44.5 | 60.9 |
| TensorRT INT8 entropy calib (C++) | 2.35 | 2.50 | 1.30 | 6.22 | 6.82 | 160.8 | 41.1 | 56.8 |
| TensorRT INT8 entropy calib (C++, optimised CPU) | 0.78 | 2.19 | 0.29 | 3.30 | 4.38 | 302.7 | 41.1 | 56.8 |
| TensorRT INT8 MinMax calib (C++) | 2.31 | 2.48 | 1.29 | 6.18 | 6.88 | 161.7 | 42.7 | 59.0 |
| TensorRT INT8 MinMax calib (C++, optimised CPU) | 0.78 | 2.20 | 0.29 | 3.30 | 4.34 | 302.6 | 42.7 | 59.0 |

## TensorRT GPU stage breakdown (P50 ms, optimised CPU path)

| Precision | H2D copy | Inference | D2H copy | GPU total |
|---|---|---|---|---|
| FP32 | 0.43 | 8.14 | 0.23 | 8.81 |
| FP16 | 0.43 | 2.50 | 0.23 | 3.16 |
| INT8 | 0.43 | 1.53 | 0.23 | 2.19 |

## Correctness checks

- **PASS**: TensorRT FP32 mAP vs PyTorch FP32: +0.10 points (expected within 0.5)
- **PASS**: TensorRT FP16 vs FP32 mAP: -0.02 points (expected almost identical)
- **CHECK**: TensorRT INT8 (entropy calibration) vs FP32 mAP: -3.43 points (a drop is expected; report it as measured)
- **PASS**: TensorRT INT8 (MinMax calibration) vs FP32 mAP: -1.79 points
- **PASS**: Optimised CPU path gives the same detections as the baseline path on 300 images (33977 vs 33977 boxes, max box diff 0.000 px)

## Headline

- TensorRT FP32 (C++, optimised): 9.99 ms/frame vs PyTorch FP32 14.48 ms -> **1.4x** end-to-end
  - CPU-path optimisation at FP32: pre 2.33 -> 0.80 ms, post 1.29 -> 0.29 ms, total 11.67 -> 9.99 ms
- TensorRT FP16 (C++, optimised): 4.25 ms/frame vs PyTorch FP32 14.48 ms -> **3.4x** end-to-end
  - CPU-path optimisation at FP16: pre 2.34 -> 0.78 ms, post 1.30 -> 0.28 ms, total 6.61 -> 4.25 ms
- TensorRT INT8 (C++, optimised): 3.30 ms/frame vs PyTorch FP32 14.48 ms -> **4.4x** end-to-end
  - CPU-path optimisation at INT8: pre 2.35 -> 0.78 ms, post 1.30 -> 0.29 ms, total 6.22 -> 3.30 ms
- Bottleneck before the CPU optimisation at FP16: CPU pre + post = 3.64 ms (55% of 6.61 ms) vs GPU 2.92 ms
- After it, at FP16: inference 2.50 ms, host-device copies 0.66 ms (15%), CPU 1.06 ms of 4.25 ms
- Bottleneck before the CPU optimisation at INT8: CPU pre + post = 3.65 ms (59% of 6.22 ms) vs GPU 2.50 ms
- After it, at INT8: inference 1.53 ms, host-device copies 0.65 ms (20%), CPU 1.06 ms of 3.30 ms

## Engines

| Precision | Size MB | Build s |
|---|---|---|
| FP32 | 56.1 | 83.3 |
| FP16 | 23.8 | 276.4 |
| INT8 (entropy calibration) | 13.9 | 765.8 |
| INT8 (minmax calibration) | 14.0 | 728.1 |
