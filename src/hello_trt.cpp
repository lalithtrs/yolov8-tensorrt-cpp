// Day-1 toolchain smoke test: if this compiles, links and prints "OK", the C++/TensorRT/CUDA
// setup is correct and the rest of the project is ordinary code.
#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <iostream>

struct Logger : nvinfer1::ILogger {
  void log(Severity s, const char* m) noexcept override {
    if (s <= Severity::kWARNING) std::cerr << "[TRT] " << m << std::endl;
  }
} gLog;

int main() {
  int rtv = 0, drv = 0;
  cudaRuntimeGetVersion(&rtv);
  cudaDriverGetVersion(&drv);
  cudaDeviceProp p{};
  if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) {
    std::cerr << "FAIL: no CUDA device visible" << std::endl;
    return 1;
  }
  std::cout << "GPU            : " << p.name << " (sm_" << p.major << p.minor << ")\n"
            << "CUDA runtime   : " << rtv << ", driver supports " << drv << "\n"
#ifdef NV_TENSORRT_MAJOR
            << "TensorRT header: " << NV_TENSORRT_MAJOR << "." << NV_TENSORRT_MINOR << "." << NV_TENSORRT_PATCH << "\n"
#endif
            << "TensorRT lib   : " << getInferLibVersion() << std::endl;
  nvinfer1::IRuntime* rt = nvinfer1::createInferRuntime(gLog);
  if (!rt) {
    std::cerr << "FAIL: createInferRuntime returned null (header/library version mismatch?)" << std::endl;
    return 1;
  }
  delete rt;
  std::cout << "OK: TensorRT runtime created from C++" << std::endl;
  return 0;
}
