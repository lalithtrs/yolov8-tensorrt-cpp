#!/usr/bin/env python3
"""Prepare and build the C++ side against the pip-installed TensorRT.

1. Finds libnvinfer.so.<major> from the pip `tensorrt` package (the same library that builds the
   engines in Python, so C++ and Python always agree on the TensorRT version).
2. Gets the matching public C++ headers (NvInfer.h ...) from the NVIDIA/TensorRT GitHub repo at the
   tag of the installed version (the pip package ships no headers).
3. Finds cuda_runtime_api.h + libcudart (system CUDA first, then pip nvidia-* packages).
4. Configures and builds with CMake; writes build/paths.json.
"""
import argparse
import glob
import json
import os
import shutil
import site
import subprocess
import sys

TRT_REPO = 'https://github.com/NVIDIA/TensorRT.git'


def run(cmd, **kw):
    print('+', ' '.join(cmd), flush=True)
    return subprocess.run(cmd, **kw)


def site_dirs():
    dirs = list(site.getsitepackages()) + [site.getusersitepackages()]
    return [d for d in dirs if os.path.isdir(d)]


def find_trt_lib():
    import tensorrt as trt
    ver = trt.__version__
    major = ver.split('.')[0]
    name = f'libnvinfer.so.{major}'
    cands = []
    try:
        import tensorrt_libs
        cands += list(getattr(tensorrt_libs, '__path__', []))
    except ImportError:
        pass
    cands.append(os.path.dirname(trt.__file__))
    for d in cands:
        if os.path.exists(os.path.join(d, name)):
            return ver, os.path.join(d, name)
    for sp in site_dirs():
        hits = sorted(glob.glob(os.path.join(sp, '**', name), recursive=True))
        if hits:
            return ver, hits[0]
    sys.exit(f'could not find {name}; is the pip tensorrt package installed?')


def get_headers(ver, dest):
    """Sparse, shallow clone of include/ at the tag that matches the installed TensorRT."""
    inc = os.path.join(dest, 'include')
    if os.path.exists(os.path.join(inc, 'NvInfer.h')):
        print('headers already present:', inc)
        return inc, open(os.path.join(dest, '.ref')).read() if os.path.exists(os.path.join(dest, '.ref')) else '?'
    p = ver.split('.')
    refs = [f'v{p[0]}.{p[1]}.{p[2]}', f'release/{p[0]}.{p[1]}', None]
    for ref in refs:
        shutil.rmtree(dest, ignore_errors=True)
        cmd = ['git', 'clone', '--depth', '1', '--filter=blob:none', '--sparse']
        if ref:
            cmd += ['-b', ref]
        r = run(cmd + [TRT_REPO, dest], capture_output=True, text=True)
        if r.returncode != 0:
            print('  not available:', ref)
            continue
        run(['git', '-C', dest, 'sparse-checkout', 'set', 'include'], check=True)
        if ref is None:
            print(f'WARNING: no tag/branch for TensorRT {ver}; using the default branch headers. '
                  'If hello_trt fails, pin tensorrt to a version with a matching GitHub tag.')
        with open(os.path.join(dest, '.ref'), 'w') as f:
            f.write(ref or 'default-branch')
        return inc, ref or 'default-branch'
    sys.exit('could not clone TensorRT headers (is Internet enabled in the notebook settings?)')


def find_cuda(major):
    """Return (include_dir, libcudart path); prefer a libcudart whose major version matches."""
    pairs = []
    for root in ['/usr/local/cuda'] + sorted(glob.glob('/usr/local/cuda-*'), reverse=True):
        for libdir in ['lib64', 'targets/x86_64-linux/lib']:
            pairs.append((os.path.join(root, 'include'), os.path.join(root, libdir)))
    for sp in site_dirs():
        for h in glob.glob(os.path.join(sp, 'nvidia', '**', 'include', 'cuda_runtime_api.h'), recursive=True):
            inc = os.path.dirname(h)
            pairs.append((inc, os.path.join(os.path.dirname(inc), 'lib')))
    fallback = None
    for inc, libdir in pairs:
        if not os.path.exists(os.path.join(inc, 'cuda_runtime_api.h')):
            continue
        exact = os.path.join(libdir, f'libcudart.so.{major}')
        if os.path.exists(exact):
            return inc, exact
        libs = sorted(glob.glob(os.path.join(libdir, 'libcudart.so*')))
        if libs and fallback is None:
            fallback = (inc, libs[0])
    if fallback:
        print(f'WARNING: no libcudart.so.{major}; using {fallback[1]}')
        return fallback
    sys.exit('cuda_runtime_api.h / libcudart not found. Try: pip install nvidia-cuda-runtime-cu12')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument('--headers-dir', default='/tmp/trt_oss')
    ap.add_argument('--cuda-major', help='default: CUDA major version of the installed torch')
    a = ap.parse_args()

    ver, trt_lib = find_trt_lib()
    if a.cuda_major is None:
        import torch
        a.cuda_major = torch.version.cuda.split('.')[0]
    trt_inc, ref = get_headers(ver, a.headers_dir)
    cuda_inc, cudart = find_cuda(a.cuda_major)
    paths = {'tensorrt': ver, 'headers_ref': ref, 'TRT_INC': trt_inc, 'TRT_LIB': trt_lib,
             'CUDA_INC': cuda_inc, 'CUDART_LIB': cudart, 'cuda_major': a.cuda_major}
    print(json.dumps(paths, indent=2))

    build = os.path.join(a.repo, 'build')
    cfg = ['cmake', '-S', a.repo, '-B', build, '-DCMAKE_BUILD_TYPE=Release'] + \
          [f'-D{k}={paths[k]}' for k in ('TRT_INC', 'TRT_LIB', 'CUDA_INC', 'CUDART_LIB')]
    if run(cfg).returncode != 0:
        sys.exit('cmake configure failed (is libopencv-dev installed?)')
    if run(['cmake', '--build', build, '-j', str(os.cpu_count() or 2)]).returncode != 0:
        sys.exit('C++ build failed; see the compiler errors above')
    with open(os.path.join(build, 'paths.json'), 'w') as f:
        json.dump(paths, f, indent=2)
    print('built:', os.path.join(build, 'hello_trt'), os.path.join(build, 'yolo_trt'))


if __name__ == '__main__':
    main()
