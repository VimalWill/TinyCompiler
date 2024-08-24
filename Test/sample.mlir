//conv2d_relu
module attributes {torch.debug_module_name = "SimpleConvModel"} {
  func.func @forward(%arg0: tensor<1x1x5x5xf32>) -> tensor<1x1x5x5xf32> {
    %cst = arith.constant dense<[[[[-0.085320197], [-0.151474833], [-0.118983984]], [[-0.0790755376], [0.154833764], [-0.0646391734]], [[-0.0562474355], [0.104069829], [0.261827648]]]]> : tensor<1x3x3x1xf32>
    %cst_0 = arith.constant dense<0.259758711> : tensor<1xf32>
    %0 = tosa.reshape %arg0 {new_shape = array<i64: 1, 5, 5, 1>} : (tensor<1x1x5x5xf32>) -> tensor<1x5x5x1xf32>
    %1 = TinyFusion.conv2d_relu %0, %cst, %cst_0 {dilation = [1, 1], max_fp = 3.40282347E+38 : f32, min_fp = 0.000000e+00 : f32, padding = [1, 1, 1, 1], stride = [1, 1]} : (tensor<1x5x5x1xf32>, tensor<1x3x3x1xf32>, tensor<1xf32>) -> tensor<1x5x5x1xf32>
    %2 = tosa.reshape %1 {new_shape = array<i64: 1, 1, 5, 5>} : (tensor<1x5x5x1xf32>) -> tensor<1x1x5x5xf32>
    return %2 : tensor<1x1x5x5xf32>
  }
}

//conv2d_silu
module attributes {torch.debug_module_name = "SimpleConvModel"} {
  func.func @forward(%arg0: tensor<1x1x5x5xf32>) -> tensor<1x1x5x5xf32> {
    %cst = arith.constant dense<[[[[-0.298598945], [0.278794706], [-0.0357439518]], [[0.188536808], [-0.315596104], [0.310037613]], [[-0.061714571], [-0.211769432], [-0.0372516736]]]]> : tensor<1x3x3x1xf32>
    %cst_0 = arith.constant dense<-0.256658554> : tensor<1xf32>
    %0 = tosa.reshape %arg0 {new_shape = array<i64: 1, 5, 5, 1>} : (tensor<1x1x5x5xf32>) -> tensor<1x5x5x1xf32>
    %1 = TinyFusion.conv2d_silu %0, %cst, %cst_0 {dilation = [1, 1], padding = [1, 1, 1, 1], stride = [1, 1]} : (tensor<1x5x5x1xf32>, tensor<1x3x3x1xf32>, tensor<1xf32>) -> tensor<1x5x5x1xf32>
    %2 = tosa.reshape %1 {new_shape = array<i64: 1, 1, 5, 5>} : (tensor<1x5x5x1xf32>) -> tensor<1x1x5x5xf32>
    return %2 : tensor<1x1x5x5xf32>
  }
}

//conv2d_lrelu
module attributes {torch.debug_module_name = "SimpleConvModel"} {
  func.func @forward(%arg0: tensor<1x1x5x5xf32>) -> tensor<1x1x5x5xf32> {
    %cst = arith.constant dense<[[[[0.329010367], [0.139047116], [0.325858593]], [[-0.0141289234], [0.236418769], [-0.019041976]], [[-0.186193079], [-0.196645662], [-0.278877497]]]]> : tensor<1x3x3x1xf32>
    %cst_0 = arith.constant dense<1.000000e-01> : tensor<1x1x1x1xf32>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<1x1x1x1xf32>
    %cst_2 = arith.constant dense<-0.167743295> : tensor<1xf32>
    %0 = tosa.reshape %arg0 {new_shape = array<i64: 1, 5, 5, 1>} : (tensor<1x1x5x5xf32>) -> tensor<1x5x5x1xf32>
    %1 = TinyFusion.conv2d_lrelu %0, %cst, %cst_2, %cst_0, %cst_1 {dilation = [1, 1], padding = [1, 1, 1, 1], stride = [1, 1]} : (tensor<1x5x5x1xf32>, tensor<1x3x3x1xf32>, tensor<1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>) -> tensor<1x5x5x1xf32>
    %2 = tosa.reshape %1 {new_shape = array<i64: 1, 1, 5, 5>} : (tensor<1x5x5x1xf32>) -> tensor<1x1x5x5xf32>
    return %2 : tensor<1x1x5x5xf32>
  }
}