# Tiny Compiler 
Minimal implementation for graph compiler with torch_mlir frontend to compile AI workloads to the NVIDIA GPU. The soul purpose of the project to understand the graph compilers and MLIR-framework. 

## focused Graph Optimization 
> More optimization will be focused in future and looking for colabs for the project and learning process

<h3>Operator fusion</h3>

A custom fusion dialect to handle hardware-independent operator fusion 
for DNN operator and the approach adopted from [TVM](https://layman-n-ish.github.io/pdfs/TVM_Review_Report.pdf).\
<b>example:</b>

```
Relay graph before fusion:
def @main(%data: Tensor[(1, 3, 64, 64), float32], %weight: Tensor[(16, 3, 3, 3), float32]) {
  %0 = nn.conv2d(%data, %weight, padding=[1, 1, 1, 1], channels=16, kernel_size=[3, 3]);
  nn.relu(%0)
}

Relay graph after fusion:
def @main(%data: Tensor[(1, 3, 64, 64), float32] /* ty=Tensor[(1, 3, 64, 64), float32] */, %weight: Tensor[(16, 3, 3, 3), float32] /* ty=Tensor[(16, 3, 3, 3), float32] */) -> Tensor[(1, 16, 64, 64), float32] {
  %1 = fn (%p0: Tensor[(1, 3, 64, 64), float32] /* ty=Tensor[(1, 3, 64, 64), float32] */, %p1: Tensor[(16, 3, 3, 3), float32] /* ty=Tensor[(16, 3, 3, 3), float32] */, Primitive=1) -> Tensor[(1, 16, 64, 64), float32] {
    %0 = nn.conv2d(%p0, %p1, padding=[1, 1, 1, 1], channels=16, kernel_size=[3, 3]) /* ty=Tensor[(1, 16, 64, 64), float32] */;
    nn.relu(%0) /* ty=Tensor[(1, 16, 64, 64), float32] */
  } /* ty=fn (Tensor[(1, 3, 64, 64), float32], Tensor[(16, 3, 3, 3), float32]) -> Tensor[(1, 16, 64, 64), float32] */;
  %1(%data, %weight) /* ty=Tensor[(1, 16, 64, 64), float32] */
}
```
from the example, the complex-out-fusable can be fused with the element-wise operator i.e., ReLu in this case to form as single operator.

