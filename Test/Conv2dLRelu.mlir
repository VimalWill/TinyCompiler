module attributes {torch.debug_module_name = "SimpleConvModel"} {
  func.func @forward(%arg0: tensor<1x1x5x5xf32>) -> tensor<1x1x5x5xf32> {
    %0 = "tosa.const"() <{value = dense<-0.167743295> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x1x1x1xf32>}> : () -> tensor<1x1x1x1xf32>
    %2 = "tosa.const"() <{value = dense<1.000000e-01> : tensor<1x1x1x1xf32>}> : () -> tensor<1x1x1x1xf32>
    %3 = "tosa.const"() <{value = dense<[[[[0.329010367], [0.139047116], [0.325858593]], [[-0.0141289234], [0.236418769], [-0.019041976]], [[-0.186193079], [-0.196645662], [-0.278877497]]]]> : tensor<1x3x3x1xf32>}> : () -> tensor<1x3x3x1xf32>
    %4 = tosa.reshape %arg0 {new_shape = array<i64: 1, 5, 5, 1>} : (tensor<1x1x5x5xf32>) -> tensor<1x5x5x1xf32>
    %5 = tosa.conv2d %4, %3, %0 {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x5x5x1xf32>, tensor<1x3x3x1xf32>, tensor<1xf32>) -> tensor<1x5x5x1xf32>
    %6 = tosa.reshape %5 {new_shape = array<i64: 1, 1, 5, 5>} : (tensor<1x5x5x1xf32>) -> tensor<1x1x5x5xf32>
    %7 = tosa.maximum %6, %1 : (tensor<1x1x5x5xf32>, tensor<1x1x1x1xf32>) -> tensor<1x1x5x5xf32>
    %8 = tosa.minimum %6, %1 : (tensor<1x1x5x5xf32>, tensor<1x1x1x1xf32>) -> tensor<1x1x5x5xf32>
    %9 = tosa.mul %8, %2 {shift = 0 : i8} : (tensor<1x1x5x5xf32>, tensor<1x1x1x1xf32>) -> tensor<1x1x5x5xf32>
    %10 = tosa.add %7, %9 : (tensor<1x1x5x5xf32>, tensor<1x1x5x5xf32>) -> tensor<1x1x5x5xf32>
    return %10 : tensor<1x1x5x5xf32>
  }
}
