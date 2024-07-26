module attributes {torch.debug_module_name = "SimpleConvModel"} {
  func.func @forward(%arg0: tensor<1x1x5x5xf32>) -> tensor<1x1x5x5xf32> {
    %0 = "tosa.const"() <{value = dense<-0.256658554> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1 = "tosa.const"() <{value = dense<[[[[-0.298598945], [0.278794706], [-0.0357439518]], [[0.188536808], [-0.315596104], [0.310037613]], [[-0.061714571], [-0.211769432], [-0.0372516736]]]]> : tensor<1x3x3x1xf32>}> : () -> tensor<1x3x3x1xf32>
    %2 = tosa.reshape %arg0 {new_shape = array<i64: 1, 5, 5, 1>} : (tensor<1x1x5x5xf32>) -> tensor<1x5x5x1xf32>
    %3 = tosa.conv2d %2, %1, %0 {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x5x5x1xf32>, tensor<1x3x3x1xf32>, tensor<1xf32>) -> tensor<1x5x5x1xf32>
    %4 = tosa.reshape %3 {new_shape = array<i64: 1, 1, 5, 5>} : (tensor<1x5x5x1xf32>) -> tensor<1x1x5x5xf32>
    %5 = tosa.sigmoid %4 : (tensor<1x1x5x5xf32>) -> tensor<1x1x5x5xf32>
    %6 = tosa.mul %5, %4 {shift = 0 : i8} : (tensor<1x1x5x5xf32>, tensor<1x1x5x5xf32>) -> tensor<1x1x5x5xf32>
    return %6 : tensor<1x1x5x5xf32>
  }
}