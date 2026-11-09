#!/usr/bin/env python3
"""Record hardware and software versions (they go into the README results section)."""
import json
import os
import platform
import subprocess
import sys


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return ''


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else 'results/env.json'
    env = {'gpu': sh('nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader -i 0'),
           'cpu': sh("grep -m1 'model name' /proc/cpuinfo | cut -d: -f2").strip(),
           'cpu_cores': os.cpu_count(), 'os': platform.platform(), 'python': platform.python_version(),
           'opencv_cpp': sh('pkg-config --modversion opencv4')}
    try:
        import torch
        env['torch'], env['torch_cuda'] = torch.__version__, torch.version.cuda
    except ImportError:
        pass
    for mod in ('tensorrt', 'ultralytics', 'cv2', 'onnx'):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            pass
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(env, f, indent=2)
    print(json.dumps(env, indent=2))


if __name__ == '__main__':
    main()
