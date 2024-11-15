# Tiny Compiler 
Minimal implementation for graph compiler with torch_mlir frontend to compile AI workloads to the NVIDIA GPU. The soul purpose of the project to understand the graph compilers and MLIR-framework. 

## focused Graph Optimization 
> More optimization will be focused in future and looking for colabs for the project and learning process

<h3>Operator fusion</h3>

<!-- A custom fusion dialect to handle hardware-independent operator fusion 
for DNN operator and the approach adopted from [TVM](https://layman-n-ish.github.io/pdfs/TVM_Review_Report.pdf).  -->

TinyFusion dialect is the part of TinyCompiler which can supports operator fusion for the [TOSA](https://mlir.llvm.org/docs/Dialects/TOSA/) instruction architecture. The fusion dialect works based on the approach discussed in the [TVM](https://layman-n-ish.github.io/pdfs/TVM_Review_Report.pdf) white-paper. TinyFusion can reduce the memory footprints in the over-all computation by avoiding the intermediate memory allocations. 

## Build Instructions
> Refer [Getting-Start](https://mlir.llvm.org/getting_started/) of MLIR to build and install MLIR & LLVM on the machine

TinyCompiler CMake Instructions:
```
$ export MLIR_DIR=~/llvm-project/build/lib/cmake/mlir
$ mkdir build && cd build 
$ cmake ..
$ make -j32
```
The compiler can be tested as follows, 
```
$ ./tools/TinyCompiler-Opt --compile ../../Test/Conv2dRelu.mlir 
```
> The pass-pipeline supports affine transformation for limited tinyfusion operators lowering most of the operators to tinyfusion, arith and tensor dialects. 

## In-progress 
1. Affine lowering for every TinyFusion operator
2. ```TinyFlow.dispatch()``` to support parallelism and scalability