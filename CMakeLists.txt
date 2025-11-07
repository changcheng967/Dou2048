# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(2048_AI_CUDA)

# 设置C++标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找CUDA
find_package(CUDA REQUIRED)

# 设置CUDA架构（针对A6000 - Ampere架构）
set(CMAKE_CUDA_ARCHITECTURES 86)

# 包含目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# 创建可执行文件
cuda_add_executable(2048_ai_cuda
    main.cu
    # 其他源文件...
)

# 设置编译选项
target_compile_options(2048_ai_cuda PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:
    --use_fast_math
    -O3
    >
)

target_compile_options(2048_ai_cuda PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:
    -O3
    -march=native
    >
)

# 链接库
target_link_libraries(2048_ai_cuda ${CUDA_LIBRARIES})
