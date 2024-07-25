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
module attributes {torch.debug_module_name = "SimpleConvModel"} {
  func.func @forward(%arg0: tensor<1x1x5x5xf32>) -> tensor<1x1x5x5xf32> {
    %0 = "tosa.const"() <{value = dense<0.259758711> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1 = "tosa.const"() <{value = dense<[[[[-0.085320197], [-0.151474833], [-0.118983984]], [[-0.0790755376], [0.154833764], [-0.0646391734]], [[-0.0562474355], [0.104069829], [0.261827648]]]]> : tensor<1x3x3x1xf32>}> : () -> tensor<1x3x3x1xf32>
    %2 = tosa.reshape %arg0 {new_shape = array<i64: 1, 5, 5, 1>} : (tensor<1x1x5x5xf32>) -> tensor<1x5x5x1xf32>
    %3 = tosa.conv2d %2, %1, %0 {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x5x5x1xf32>, tensor<1x3x3x1xf32>, tensor<1xf32>) -> tensor<1x5x5x1xf32>
    %4 = tosa.reshape %3 {new_shape = array<i64: 1, 1, 5, 5>} : (tensor<1x5x5x1xf32>) -> tensor<1x1x5x5xf32>
    %5 = tosa.clamp %4 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x1x5x5xf32>) -> tensor<1x1x5x5xf32>
    return %5 : tensor<1x1x5x5xf32>
  }
}


Relay graph after fusion:
module attributes {torch.debug_module_name = "SimpleConvModel"} {
  func.func @forward(%arg0: tensor<1x1x5x5xf32>) -> tensor<1x1x5x5xf32> {
    %0 = "tosa.const"() <{value = dense<0.259758711> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1 = "tosa.const"() <{value = dense<[[[[-0.085320197], [-0.151474833], [-0.118983984]], [[-0.0790755376], [0.154833764], [-0.0646391734]], [[-0.0562474355], [0.104069829], [0.261827648]]]]> : tensor<1x3x3x1xf32>}> : () -> tensor<1x3x3x1xf32>
    %2 = tosa.reshape %arg0 {new_shape = array<i64: 1, 5, 5, 1>} : (tensor<1x1x5x5xf32>) -> tensor<1x5x5x1xf32>
    %3 = "TinyFusion.conv2d_relu"(%2, %1, %0) <{dilation = [1, 1], max_fp = 3.40282347E+38 : f32, min_fp = 0.000000e+00 : f32, padding = [1, 1, 1, 1], stride = [1, 1]}> : (tensor<1x5x5x1xf32>, tensor<1x3x3x1xf32>, tensor<1xf32>) -> tensor<1x5x5x1xf32>
    %4 = tosa.reshape %3 {new_shape = array<i64: 1, 1, 5, 5>} : (tensor<1x5x5x1xf32>) -> tensor<1x1x5x5xf32>
    return %4 : tensor<1x1x5x5xf32>
  }
}

```
The above example holds "TinyFusion.conv2d_relu" operator is a fused version of "tosa.conv2d" and "tosa.clamp" operator. 



