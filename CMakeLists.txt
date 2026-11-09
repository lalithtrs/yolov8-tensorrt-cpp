cmake_minimum_required(VERSION 3.18)
project(yolo_trt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Paths are passed in by scripts/setup_cpp.py (it finds them on the machine):
#   TRT_INC    folder with NvInfer.h            TRT_LIB     full path to libnvinfer.so.<major>
#   CUDA_INC   folder with cuda_runtime_api.h   CUDART_LIB  full path to libcudart.so[.<major>]
foreach(v TRT_INC TRT_LIB CUDA_INC CUDART_LIB)
  if(NOT DEFINED ${v})
    message(FATAL_ERROR "Pass -D${v}=... (run scripts/setup_cpp.py instead of calling cmake directly)")
  endif()
endforeach()

get_filename_component(TRT_LIB_DIR ${TRT_LIB} DIRECTORY)
get_filename_component(CUDART_DIR ${CUDART_LIB} DIRECTORY)
# Bake the library folders into the binaries so no LD_LIBRARY_PATH is needed. The C++ app then
# loads the very same libnvinfer that built the engines in Python, so versions always match.
set(CMAKE_BUILD_RPATH "${TRT_LIB_DIR};${CUDART_DIR}")

add_executable(hello_trt src/hello_trt.cpp)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio)
add_executable(yolo_trt src/main.cpp)
target_include_directories(yolo_trt PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(yolo_trt PRIVATE ${OpenCV_LIBS})
target_compile_options(yolo_trt PRIVATE -O3 -march=native)

foreach(t hello_trt yolo_trt)
  target_include_directories(${t} PRIVATE ${TRT_INC} ${CUDA_INC})
  target_link_libraries(${t} PRIVATE ${TRT_LIB} ${CUDART_LIB})
  target_compile_options(${t} PRIVATE -Wall -Wno-deprecated-declarations)
endforeach()
