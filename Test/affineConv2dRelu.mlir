#map = affine_map<(d0, d1) -> (d0 + d1)>
module attributes {torch.debug_module_name = "SimpleConvModel"} {
  func.func @forward(%arg0: tensor<1x1x5x5xf32>) -> tensor<1x1x5x5xf32> {
    %cst = arith.constant dense<[[[[-0.085320197], [-0.151474833], [-0.118983984]], [[-0.0790755376], [0.154833764], [-0.0646391734]], [[-0.0562474355], [0.104069829], [0.261827648]]]]> : tensor<1x3x3x1xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 0.259758711 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %alloc = memref.alloc() : memref<1x5x5x1xf32>
    %alloc_2 = memref.alloc() : memref<1xf32>
    %alloc_3 = memref.alloc() : memref<1x3x3x1xf32>
    %collapsed = tensor.collapse_shape %arg0 [[0, 1], [2], [3]] : tensor<1x1x5x5xf32> into tensor<1x5x5xf32>
    %expanded = tensor.expand_shape %collapsed [[0], [1], [2, 3]] output_shape [1, 5, 5, 1] : tensor<1x5x5xf32> into tensor<1x5x5x1xf32>
    %padded = tensor.pad %expanded low[%c0, %c0, %c1, %c1] high[%c0, %c0, %c1, %c1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x5x5x1xf32> to tensor<1x6x6x1xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 3 {
        affine.for %arg3 = 0 to 3 {
          affine.for %arg4 = 0 to 1 {
            %extracted = tensor.extract %cst[%arg1, %arg2, %arg3, %arg4] : tensor<1x3x3x1xf32>
            memref.store %extracted, %alloc_3[%arg1, %arg2, %arg3, %arg4] : memref<1x3x3x1xf32>
          }
        }
      }
    }
    affine.for %arg1 = 0 to 1 {
      memref.store %cst_1, %alloc_2[%arg1] : memref<1xf32>
    }
    %alloc_4 = memref.alloc() : memref<f32>
    memref.store %cst_0, %alloc_4[] : memref<f32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 5 {
        affine.for %arg3 = 0 to 5 {
          affine.for %arg4 = 0 to 1 {
            memref.store %cst_0, %alloc[%arg1, %arg2, %arg3, %arg4] : memref<1x5x5x1xf32>
          }
        }
      }
    }
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 5 {
        affine.for %arg3 = 0 to 5 {
          affine.for %arg4 = 0 to 1 {
            affine.for %arg5 = 0 to 3 {
              affine.for %arg6 = 0 to 3 {
                affine.for %arg7 = 0 to 1 {
                  %1 = affine.apply #map(%arg2, %arg5)
                  %2 = affine.apply #map(%arg3, %arg6)
                  %extracted = tensor.extract %padded[%arg1, %1, %2, %arg7] : tensor<1x6x6x1xf32>
                  %3 = memref.load %alloc_3[%arg5, %arg6, %arg7, %arg4] : memref<1x3x3x1xf32>
                  %4 = arith.mulf %extracted, %3 : f32
                  %5 = memref.load %alloc[%arg1, %arg2, %arg3, %arg4] : memref<1x5x5x1xf32>
                  %6 = arith.addf %4, %5 : f32
                  %7 = memref.load %alloc_2[%arg4] : memref<1xf32>
                  %8 = arith.addf %6, %7 : f32
                  %9 = memref.load %alloc_4[] : memref<f32>
                  %10 = arith.maximumf %9, %8 : f32
                  memref.store %10, %alloc[%arg1, %arg2, %arg3, %arg4] : memref<1x5x5x1xf32>
                }
              }
            }
          }
        }
      }
    }
    %0 = bufferization.to_tensor %alloc : memref<1x5x5x1xf32>
    %collapsed_5 = tensor.collapse_shape %0 [[0], [1], [2, 3]] : tensor<1x5x5x1xf32> into tensor<1x5x5xf32>
    %expanded_6 = tensor.expand_shape %collapsed_5 [[0, 1], [2], [3]] output_shape [1, 1, 5, 5] : tensor<1x5x5xf32> into tensor<1x1x5x5xf32>
    memref.dealloc %alloc_3 : memref<1x3x3x1xf32>
    memref.dealloc %alloc_2 : memref<1xf32>
    memref.dealloc %alloc_4 : memref<f32>
    memref.dealloc %alloc : memref<1x5x5x1xf32>
    return %expanded_6 : tensor<1x1x5x5xf32>
  }
}
