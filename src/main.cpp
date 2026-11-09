// yolo_trt: YOLOv8 inference with TensorRT in C++.
//
//   yolo_trt ENGINE bench LIST            [--fast] [--iters N] [--warmup N]
//   yolo_trt ENGINE coco  LIST OUT.csv    [--fast]
//   yolo_trt ENGINE video IN.mp4 OUT.mp4  [--fast]
//
// Two CPU code paths, selected with --fast:
//   baseline : OpenCV letterbox + cvtColor + convertTo + split; anchor-major decode (strided reads)
//   --fast   : one fused pass (resize -> RGB/255/CHW + padding, LUT);  class-major decode (contiguous,
//              vectorisable reads). Both paths give the same detections (checked by run_trt.py).
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#define CK(x)                                                                                   \
  do {                                                                                          \
    cudaError_t e_ = (x);                                                                       \
    if (e_ != cudaSuccess) {                                                                    \
      std::cerr << "CUDA error: " << cudaGetErrorString(e_) << " at line " << __LINE__ << "\n"; \
      std::exit(1);                                                                             \
    }                                                                                           \
  } while (0)

[[noreturn]] static void die(const std::string& m) { std::cerr << "ERROR: " << m << std::endl; std::exit(1); }

struct Logger : nvinfer1::ILogger {
  void log(Severity s, const char* m) noexcept override {
    if (s <= Severity::kWARNING) std::cerr << "[TRT] " << m << std::endl;
  }
} gLog;

static const int COCO91[80] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                               22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                               46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
                               67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90};

static const char* NAMES[80] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"};

struct Det { float x1, y1, x2, y2, score; int cls; };
struct LB { float scale; int px, py, nw, nh; };
struct GpuTimes { float h2d, infer, d2h, total; };

// ---------------------------------------------------------------------------------------------
// TensorRT engine wrapper: one static [1,3,S,S] float input, one [1,C,N] float output.
// Pinned host buffers + one CUDA stream; CUDA events time each GPU stage.
// ---------------------------------------------------------------------------------------------
struct Trt {
  std::unique_ptr<nvinfer1::IRuntime> rt;
  std::unique_ptr<nvinfer1::ICudaEngine> eng;
  std::unique_ptr<nvinfer1::IExecutionContext> ctx;
  int S = 640, C = 84, N = 8400;
  size_t inBytes = 0, outBytes = 0;
  float *dIn = nullptr, *dOut = nullptr, *hIn = nullptr, *hOut = nullptr;
  cudaStream_t st = nullptr;
  cudaEvent_t ev[4] = {};
  std::string inName, outName;

  explicit Trt(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) die("cannot open engine " + path);
    std::vector<char> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    rt.reset(nvinfer1::createInferRuntime(gLog));
    if (!rt) die("createInferRuntime failed");
    eng.reset(rt->deserializeCudaEngine(buf.data(), buf.size()));
    if (!eng) die("engine load failed (built with another TensorRT version or GPU? rebuild it in this session)");
    ctx.reset(eng->createExecutionContext());
    if (!ctx) die("createExecutionContext failed");

    int nIn = 0, nOut = 0;
    for (int i = 0; i < eng->getNbIOTensors(); ++i) {
      const char* n = eng->getIOTensorName(i);
      if (eng->getTensorIOMode(n) == nvinfer1::TensorIOMode::kINPUT) { inName = n; ++nIn; }
      else { outName = n; ++nOut; }
    }
    if (nIn != 1 || nOut != 1) die("expected exactly 1 input and 1 output tensor");
    if (eng->getTensorDataType(inName.c_str()) != nvinfer1::DataType::kFLOAT ||
        eng->getTensorDataType(outName.c_str()) != nvinfer1::DataType::kFLOAT)
      die("expected float32 input/output tensors");

    auto id = eng->getTensorShape(inName.c_str());
    if (id.nbDims != 4 || id.d[0] != 1 || id.d[1] != 3 || id.d[2] != id.d[3])
      die("expected a static [1,3,S,S] input (export ONNX with dynamic=False, batch=1)");
    S = (int)id.d[2];
    auto od = eng->getTensorShape(outName.c_str());
    if (od.nbDims != 3 || od.d[0] != 1) die("expected a [1,C,N] output");
    C = (int)od.d[1];
    N = (int)od.d[2];

    inBytes = 3ull * S * S * sizeof(float);
    outBytes = (size_t)C * N * sizeof(float);
    CK(cudaMalloc((void**)&dIn, inBytes));
    CK(cudaMalloc((void**)&dOut, outBytes));
    CK(cudaMallocHost((void**)&hIn, inBytes));
    CK(cudaMallocHost((void**)&hOut, outBytes));
    CK(cudaStreamCreate(&st));
    for (auto& e : ev) CK(cudaEventCreate(&e));
    if (!ctx->setTensorAddress(inName.c_str(), dIn) || !ctx->setTensorAddress(outName.c_str(), dOut))
      die("setTensorAddress failed");
    std::cerr << "engine " << path << ": input " << inName << " [1,3," << S << "," << S << "], output "
              << outName << " [1," << C << "," << N << "]" << std::endl;
  }

  ~Trt() {
    for (auto& e : ev) if (e) cudaEventDestroy(e);
    if (st) cudaStreamDestroy(st);
    cudaFree(dIn); cudaFree(dOut); cudaFreeHost(hIn); cudaFreeHost(hOut);
  }

  // Host->device copy, inference, device->host copy on one stream. Blocks until done.
  GpuTimes gpu() {
    CK(cudaEventRecord(ev[0], st));
    CK(cudaMemcpyAsync(dIn, hIn, inBytes, cudaMemcpyHostToDevice, st));
    CK(cudaEventRecord(ev[1], st));
    if (!ctx->enqueueV3(st)) die("enqueueV3 failed");
    CK(cudaEventRecord(ev[2], st));
    CK(cudaMemcpyAsync(hOut, dOut, outBytes, cudaMemcpyDeviceToHost, st));
    CK(cudaEventRecord(ev[3], st));
    CK(cudaStreamSynchronize(st));
    GpuTimes t{};
    CK(cudaEventElapsedTime(&t.h2d, ev[0], ev[1]));
    CK(cudaEventElapsedTime(&t.infer, ev[1], ev[2]));
    CK(cudaEventElapsedTime(&t.d2h, ev[2], ev[3]));
    CK(cudaEventElapsedTime(&t.total, ev[0], ev[3]));
    return t;
  }
};

// ---------------------------------------------------------------------------------------------
// Preprocessing: letterbox to SxS (grey 114 padding, centred), BGR->RGB, /255, HWC->CHW.
// Must match scripts/build_engine.py:prep() (used for INT8 calibration).
// ---------------------------------------------------------------------------------------------
static void letterboxGeom(const cv::Mat& img, int S, LB& lb) {
  lb.scale = std::min(S / (float)img.cols, S / (float)img.rows);
  lb.nw = (int)std::round(img.cols * lb.scale);
  lb.nh = (int)std::round(img.rows * lb.scale);
  lb.px = (S - lb.nw) / 2;
  lb.py = (S - lb.nh) / 2;
}

// Baseline: straightforward OpenCV calls (several full-image passes and temporaries).
static void preprocessBaseline(const cv::Mat& bgr, int S, float* dst, LB& lb) {
  letterboxGeom(bgr, S, lb);
  cv::Mat r, canvas(S, S, CV_8UC3, cv::Scalar(114, 114, 114));
  cv::resize(bgr, r, cv::Size(lb.nw, lb.nh));
  r.copyTo(canvas(cv::Rect(lb.px, lb.py, lb.nw, lb.nh)));
  cv::cvtColor(canvas, canvas, cv::COLOR_BGR2RGB);
  cv::Mat f;
  canvas.convertTo(f, CV_32FC3, 1.0 / 255);
  std::vector<cv::Mat> ch = {cv::Mat(S, S, CV_32F, dst), cv::Mat(S, S, CV_32F, dst + S * S),
                             cv::Mat(S, S, CV_32F, dst + 2 * S * S)};
  cv::split(f, ch);
}

// Fast: resize, then ONE pass that writes RGB planes (CHW) and padding directly into the
// pinned input buffer. A 256-entry LUT replaces the per-pixel multiply by 1/255.
static void preprocessFast(const cv::Mat& bgr, int S, float* dst, LB& lb, cv::Mat& resized) {
  static float lut[256];
  static bool init = false;
  if (!init) { for (int i = 0; i < 256; ++i) lut[i] = (float)i * (float)(1.0 / 255); init = true; }
  const float pad = lut[114];
  letterboxGeom(bgr, S, lb);
  cv::resize(bgr, resized, cv::Size(lb.nw, lb.nh));
  float* R = dst;
  float* G = dst + (size_t)S * S;
  float* B = dst + 2 * (size_t)S * S;
  for (int y = 0; y < S; ++y) {
    float *r = R + (size_t)y * S, *g = G + (size_t)y * S, *b = B + (size_t)y * S;
    if (y < lb.py || y >= lb.py + lb.nh) {
      std::fill(r, r + S, pad); std::fill(g, g + S, pad); std::fill(b, b + S, pad);
      continue;
    }
    std::fill(r, r + lb.px, pad); std::fill(g, g + lb.px, pad); std::fill(b, b + lb.px, pad);
    const uint8_t* p = resized.ptr<uint8_t>(y - lb.py);
    for (int x = 0; x < lb.nw; ++x) {
      b[lb.px + x] = lut[p[3 * x + 0]];
      g[lb.px + x] = lut[p[3 * x + 1]];
      r[lb.px + x] = lut[p[3 * x + 2]];
    }
    const int e = lb.px + lb.nw;
    std::fill(r + e, r + S, pad); std::fill(g + e, g + S, pad); std::fill(b + e, b + S, pad);
  }
}

// ---------------------------------------------------------------------------------------------
// Postprocessing. YOLOv8 output is [C=84, N=8400] row-major: rows 0-3 = cx,cy,w,h; rows 4..83 =
// class scores (already sigmoid-ed, no objectness). Class-aware greedy NMS, max 300 detections.
// ---------------------------------------------------------------------------------------------
static float iou(const Det& a, const Det& b) {
  float ix = std::max(0.f, std::min(a.x2, b.x2) - std::max(a.x1, b.x1));
  float iy = std::max(0.f, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
  float inter = ix * iy;
  float u = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
  return u > 0 ? inter / u : 0.f;
}

static Det decodeBox(const float* o, int N, int i, float score, int cls, const LB& lb, int W, int H) {
  float cx = o[i], cy = o[N + i], w = o[2 * N + i], h = o[3 * N + i];
  Det d;
  d.x1 = std::clamp((cx - w / 2 - lb.px) / lb.scale, 0.f, (float)W);
  d.y1 = std::clamp((cy - h / 2 - lb.py) / lb.scale, 0.f, (float)H);
  d.x2 = std::clamp((cx + w / 2 - lb.px) / lb.scale, 0.f, (float)W);
  d.y2 = std::clamp((cy + h / 2 - lb.py) / lb.scale, 0.f, (float)H);
  d.score = score;
  d.cls = cls;
  return d;
}

static std::vector<Det> nms(std::vector<Det>& cand, float nmsT, size_t maxDet = 300) {
  std::stable_sort(cand.begin(), cand.end(), [](const Det& a, const Det& b) { return a.score > b.score; });
  std::vector<Det> keep;
  for (const Det& d : cand) {
    bool ok = true;
    for (const Det& k : keep)
      if (k.cls == d.cls && iou(k, d) > nmsT) { ok = false; break; }
    if (ok) keep.push_back(d);
    if (keep.size() >= maxDet) break;
  }
  return keep;
}

// Baseline: for every anchor, walk its 80 class scores with stride N (cache-unfriendly).
static std::vector<Det> postprocessBaseline(const float* o, int C, int N, const LB& lb, int W, int H,
                                            float conf, float nmsT) {
  std::vector<Det> cand;
  for (int i = 0; i < N; ++i) {
    int best = 0;
    float bs = 0;
    for (int k = 4; k < C; ++k) {
      float s = o[(size_t)k * N + i];
      if (s > bs) { bs = s; best = k - 4; }
    }
    if (bs < conf) continue;
    cand.push_back(decodeBox(o, N, i, bs, best, lb, W, H));
  }
  return nms(cand, nmsT);
}

// Fast: walk each class row contiguously and keep a running max per anchor (auto-vectorised).
// Same comparison order (k ascending, strict >) as the baseline, so the chosen class is the same.
struct FastPost {
  std::vector<float> best;
  std::vector<int> bcls;
  std::vector<Det> cand;
  std::vector<Det> run(const float* o, int C, int N, const LB& lb, int W, int H, float conf, float nmsT) {
    best.assign(N, 0.f);
    bcls.assign(N, 0);
    float* bp = best.data();
    int* cp = bcls.data();
    for (int k = 4; k < C; ++k) {
      const float* row = o + (size_t)k * N;
      const int cls = k - 4;
      for (int i = 0; i < N; ++i) {
        const bool gt = row[i] > bp[i];
        bp[i] = gt ? row[i] : bp[i];
        cp[i] = gt ? cls : cp[i];
      }
    }
    cand.clear();
    for (int i = 0; i < N; ++i)
      if (bp[i] >= conf) cand.push_back(decodeBox(o, N, i, bp[i], cp[i], lb, W, H));
    return nms(cand, nmsT);
  }
};

struct Pipeline {
  Trt& t;
  bool fast;
  cv::Mat resized;
  FastPost fp;
  void pre(const cv::Mat& im, LB& lb) {
    if (fast) preprocessFast(im, t.S, t.hIn, lb, resized);
    else preprocessBaseline(im, t.S, t.hIn, lb);
  }
  std::vector<Det> post(const LB& lb, int W, int H, float conf, float nmsT) {
    if (fast) return fp.run(t.hOut, t.C, t.N, lb, W, H, conf, nmsT);
    return postprocessBaseline(t.hOut, t.C, t.N, lb, W, H, conf, nmsT);
  }
};

// ---------------------------------------------------------------------------------------------
using Clk = std::chrono::steady_clock;
static double msd(Clk::time_point a, Clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }
static double pct(std::vector<double> v, double p) {
  std::sort(v.begin(), v.end());
  return v[(size_t)std::llround(p * (v.size() - 1))];
}
static std::vector<std::string> readList(const std::string& p) {
  std::ifstream f(p);
  if (!f) die("cannot open list " + p);
  std::vector<std::string> v;
  std::string l;
  while (std::getline(f, l)) {
    while (!l.empty() && (l.back() == '\r' || l.back() == ' ')) l.pop_back();
    if (!l.empty()) v.push_back(l);
  }
  return v;
}
static int imageId(const std::string& p) {
  size_t s = p.find_last_of('/'), d = p.find_last_of('.');
  return std::stoi(p.substr(s + 1, d - s - 1));
}

[[noreturn]] static void usage() {
  std::cerr << "usage: yolo_trt ENGINE bench LIST [--fast] [--iters N] [--warmup N]\n"
               "       yolo_trt ENGINE coco LIST OUT.csv [--fast]\n"
               "       yolo_trt ENGINE video IN.mp4 OUT.mp4 [--fast]\n";
  std::exit(1);
}

int main(int argc, char** argv) {
  std::vector<std::string> pos;
  bool fast = false;
  int iters = 500, warmup = 50;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--fast") fast = true;
    else if (a == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
    else if (a == "--warmup" && i + 1 < argc) warmup = std::atoi(argv[++i]);
    else if (a.rfind("--", 0) == 0) usage();
    else pos.push_back(a);
  }
  if (pos.size() < 3) usage();
  const std::string mode = pos[1];
  Trt t(pos[0]);
  Pipeline P{t, fast, {}, {}};

  if (mode == "bench") {
    std::vector<cv::Mat> imgs;
    for (auto& p : readList(pos[2])) {
      cv::Mat im = cv::imread(p);
      if (!im.empty()) imgs.push_back(im);
    }
    if (imgs.empty()) die("no readable images in " + pos[2]);
    std::vector<double> pre, h2d, inf, d2h, gpu, post, tot;
    size_t ndet = 0;
    for (int i = 0; i < warmup + iters; ++i) {
      const cv::Mat& im = imgs[i % imgs.size()];
      LB lb;
      auto a = Clk::now();
      P.pre(im, lb);
      auto b = Clk::now();
      GpuTimes g = t.gpu();
      auto c = Clk::now();
      auto dets = P.post(lb, im.cols, im.rows, 0.25f, 0.45f);
      auto e = Clk::now();
      if (i >= warmup) {
        pre.push_back(msd(a, b)); post.push_back(msd(c, e)); tot.push_back(msd(a, e));
        h2d.push_back(g.h2d); inf.push_back(g.infer); d2h.push_back(g.d2h); gpu.push_back(g.total);
        ndet += dets.size();
      }
    }
    auto rep = [](const char* n, const std::vector<double>& v) {
      std::cout << n << " P50 " << std::fixed << std::setprecision(3) << pct(v, 0.5) << " ms   P95 " << pct(v, 0.95)
                << " ms" << std::endl;
    };
    std::cout << "path: " << (fast ? "fast" : "baseline") << ", images: " << imgs.size() << ", warmup " << warmup
              << ", timed " << iters << std::endl;
    rep("preprocess (CPU)      ", pre);
    rep("  H2D copy (GPU)      ", h2d);
    rep("  inference (GPU)     ", inf);
    rep("  D2H copy (GPU)      ", d2h);
    rep("gpu total             ", gpu);
    rep("postprocess (CPU)     ", post);
    rep("total per frame       ", tot);
    std::cout << "FPS (1000 / total P50): " << 1000.0 / pct(tot, 0.5) << std::endl;
    std::cout << std::setprecision(4) << "JSON {\"path\":\"" << (fast ? "fast" : "baseline") << "\""
              << ",\"pre_p50\":" << pct(pre, .5) << ",\"pre_p95\":" << pct(pre, .95)
              << ",\"h2d_p50\":" << pct(h2d, .5) << ",\"infer_p50\":" << pct(inf, .5) << ",\"infer_p95\":" << pct(inf, .95)
              << ",\"d2h_p50\":" << pct(d2h, .5) << ",\"gpu_p50\":" << pct(gpu, .5) << ",\"gpu_p95\":" << pct(gpu, .95)
              << ",\"post_p50\":" << pct(post, .5) << ",\"post_p95\":" << pct(post, .95)
              << ",\"total_p50\":" << pct(tot, .5) << ",\"total_p95\":" << pct(tot, .95)
              << ",\"fps\":" << 1000.0 / pct(tot, .5) << ",\"avg_dets\":" << (double)ndet / iters << "}" << std::endl;
    return 0;
  }

  if (mode == "coco") {
    if (pos.size() < 4) usage();
    std::ofstream out(pos[3]);
    if (!out) die("cannot write " + pos[3]);
    out << std::fixed;
    size_t n = 0;
    for (auto& p : readList(pos[2])) {
      cv::Mat im = cv::imread(p);
      if (im.empty()) { std::cerr << "skip unreadable " << p << std::endl; continue; }
      LB lb;
      P.pre(im, lb);
      t.gpu();
      int id = imageId(p);
      for (auto& q : P.post(lb, im.cols, im.rows, 0.001f, 0.7f))
        out << id << ',' << COCO91[q.cls] << ',' << std::setprecision(2) << q.x1 << ',' << q.y1 << ','
            << (q.x2 - q.x1) << ',' << (q.y2 - q.y1) << ',' << std::setprecision(5) << q.score << '\n';
      if (++n % 500 == 0) std::cerr << n << " images" << std::endl;
    }
    std::cout << "wrote detections for " << n << " images to " << pos[3] << std::endl;
    return 0;
  }

  if (mode == "video") {
    if (pos.size() < 4) usage();
    cv::VideoCapture cap(pos[2]);
    if (!cap.isOpened()) die("cannot open video " + pos[2]);
    double fps = cap.get(cv::CAP_PROP_FPS);
    cv::Size sz((int)cap.get(cv::CAP_PROP_FRAME_WIDTH), (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    cv::VideoWriter w(pos[3], cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps > 0 ? fps : 30, sz);
    if (!w.isOpened()) die("cannot write video " + pos[3]);
    cv::Mat f;
    int n = 0;
    std::vector<double> ft;
    auto s0 = Clk::now();
    while (cap.read(f)) {
      auto a = Clk::now();
      LB lb;
      P.pre(f, lb);
      t.gpu();
      auto dets = P.post(lb, f.cols, f.rows, 0.25f, 0.45f);
      ft.push_back(msd(a, Clk::now()));
      for (auto& d : dets) {
        cv::Point p1((int)d.x1, (int)d.y1), p2((int)d.x2, (int)d.y2);
        cv::rectangle(f, p1, p2, cv::Scalar(0, 200, 0), 2);
        char lab[64];
        std::snprintf(lab, sizeof(lab), "%s %.2f", NAMES[d.cls], d.score);
        int base = 0;
        cv::Size ts = cv::getTextSize(lab, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base);
        int ty = std::max(p1.y, ts.height + 4);
        cv::rectangle(f, cv::Point(p1.x, ty - ts.height - 4), cv::Point(p1.x + ts.width + 4, ty), cv::Scalar(0, 200, 0), -1);
        cv::putText(f, lab, cv::Point(p1.x + 2, ty - 3), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
      }
      w.write(f);
      ++n;
    }
    if (n == 0) die("no frames read from " + pos[2]);
    std::cout << n << " frames, pipeline P50 " << pct(ft, 0.5) << " ms/frame (" << 1000.0 / pct(ft, 0.5)
              << " FPS excl. video decode/encode), wall " << n * 1000.0 / msd(s0, Clk::now()) << " FPS incl. I/O"
              << std::endl;
    return 0;
  }
  usage();
}
