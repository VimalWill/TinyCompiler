# Tiny Compiler
Minimal implementation for a compiler to compile graph / deep learning workloads for Edge GPUs

## Workflow
The stages of the compiler includes,
1. Frontend (currently torch_mlir)
2. MidEnd   (MLIR-style of optimization pass)
3. Backend  (Custom Code-gen dialect for GPUs)
