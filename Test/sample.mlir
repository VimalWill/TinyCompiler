module attributes {torch.debug_module_name = "SimpleConvModel"} {
  func.func @forward(%arg0: tensor<1x1x5x5xf32>) -> tensor<1x1x5x5xf32> {
    %0 = "tosa.const"() <{value = dense<0.259758711> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1 = "tosa.const"() <{value = dense<[[[[-0.085320197], [-0.151474833], [-0.118983984]], [[-0.0790755376], [0.154833764], [-0.0646391734]], [[-0.0562474355], [0.104069829], [0.261827648]]]]> : tensor<1x3x3x1xf32>}> : () -> tensor<1x3x3x1xf32>
    %2 = tosa.reshape %arg0 {new_shape = array<i64: 1, 5, 5, 1>} : (tensor<1x1x5x5xf32>) -> tensor<1x5x5x1xf32>
    %3 = "TinyFusion.conv2d_relu"(%2, %1, %0) {dilation = [1, 1], padding = [1, 1, 1, 1], stride = [1, 1], max_fp = 6.0 : f32, min_fp = 0.0 : f32} : (tensor<1x5x5x1xf32>, tensor<1x3x3x1xf32>, tensor<1xf32>) -> tensor<1x5x5x1xf32>
    %4 = tosa.reshape %3 {new_shape = array<i64: 1, 1, 5, 5>} : (tensor<1x5x5x1xf32>) -> tensor<1x1x5x5xf32>
    return %4 : tensor<1x1x5x5xf32>
  }
}
