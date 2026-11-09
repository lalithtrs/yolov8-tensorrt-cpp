#!/usr/bin/env python3
"""Demo: run the C++ app in video mode and make a GIF for the README.

Without --video, a short slideshow clip is made from COCO images that were NOT used for mAP
(drop in any real .mp4 with --video for a nicer demo).
"""
import argparse
import os
import subprocess
import sys

import cv2
import numpy as np


def slideshow(files, out, size=(1280, 720), fps=2):
    W, H = size
    w = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for p in files:
        im = cv2.imread(p)
        if im is None:
            continue
        s = min(W / im.shape[1], H / im.shape[0])
        nw, nh = int(im.shape[1] * s), int(im.shape[0] * s)
        frame = np.full((H, W, 3), 32, np.uint8)
        frame[(H - nh) // 2:(H - nh) // 2 + nh, (W - nw) // 2:(W - nw) // 2 + nw] = cv2.resize(im, (nw, nh))
        w.write(frame)
    w.release()


def to_gif(video, out, max_frames=24, width=640, ms=500):
    from PIL import Image
    cap = cv2.VideoCapture(video)
    frames = []
    while len(frames) < max_frames:
        ok, f = cap.read()
        if not ok:
            break
        h = int(f.shape[0] * width / f.shape[1])
        frames.append(Image.fromarray(cv2.cvtColor(cv2.resize(f, (width, h)), cv2.COLOR_BGR2RGB)))
    if not frames:
        sys.exit('no frames in ' + video)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=ms, loop=0, optimize=True)
    print('saved', out, len(frames), 'frames')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bin', required=True)
    ap.add_argument('--engine', required=True)
    ap.add_argument('--list', help='image list to build a slideshow from (e.g. calib.txt)')
    ap.add_argument('--video', help='use this video instead of a slideshow')
    ap.add_argument('--n', type=int, default=24)
    ap.add_argument('--out-dir', required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    src = a.video
    if not src:
        files = [l.strip() for l in open(a.list) if l.strip()][:a.n]
        src = os.path.join(a.out_dir, 'demo_input.mp4')
        slideshow(files, src)
    out = os.path.join(a.out_dir, 'demo_output.mp4')
    subprocess.run([a.bin, a.engine, 'video', src, out, '--fast'], check=True)
    to_gif(out, os.path.join(a.out_dir, 'demo.gif'), ms=500 if not a.video else 66)


if __name__ == '__main__':
    main()
