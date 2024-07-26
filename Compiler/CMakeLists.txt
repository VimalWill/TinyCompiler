cmake_minimum_required(VERSION 3.15)

project(TinyFusion)

set(CMAKE_CXX_STANDARD 17)

# if (UNIX AND NOT APPLE)
# #    set(MLIR_DIR "/home/lay/llvm/build/lib/cmake/mlir")
# #    set(LLVM_DIR "/home/lay/llvm/build/lib/cmake/llvm")
#     set(MLIR_DIR "/mnt/c/soft/llvm+mlir-17.0.0-x86_64-linux-gnu-ubuntu-18.04-release/lib/cmake/mlir")
#     set(LLVM_DIR "/mnt/c/soft/llvm+mlir-17.0.0-x86_64-linux-gnu-ubuntu-18.04-release/lib/cmake/llvm")
# elseif (UNIX AND APPLE)
#     set(MLIR_DIR "/Users/lei/soft/llvm-project/build/lib/cmake/mlir")
#     set(LLVM_DIR "/Users/lei/soft/llvm-project/build/lib/cmake/llvm")
# endif ()

find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

set(FLOW_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(FLOW_BINARY_DIR ${PROJECT_BINARY_DIR})

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${FLOW_SOURCE_DIR}/includes)
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(includes)
add_subdirectory(src)
add_subdirectory(tools)