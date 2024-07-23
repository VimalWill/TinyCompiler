#ifndef _COMPILER_INCLUDES_DIALECT_TINYFUSIONOPS_TD_
#define _COMPILER_INCLUDES_DIALECT_TINYFUSIONOPS_TD_

include "Dialect/TinyFusionDialect.td"

include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

// Define the TinyFusion operation base class
class Tiny_Fusion_Op<string mnemonic, list<Trait> traits = []> : Op<Tiny_Fusion_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Conv2dReluOp
//===----------------------------------------------------------------------===//
def Conv2dReluOp : Tiny_Fusion_Op<"conv2d_relu", [Pure]> {
    let summary = "Conv2D followed by ReLU fusion operation";
    let description = [{
        Fuse a Conv2D operation with a ReLU activation function.
        ```
            TinyFusion.conv2d_relu
        ```
    }];

    // Define the operation arguments
    let arguments = (ins
        TensorOf<[F32]>:$input_tensor,
        TensorOf<[F32]>:$weight_tensor,
        Optional<TensorOf<[F32]>>:$bias_tensor,
        ArrayAttr:$dilation, 
        ArrayAttr:$padding,
        ArrayAttr:$stride,
        F32Attr:$max_fp,
        F32Attr:$min_fp
    );

    // Define the operation results
    let results = (outs TensorOf<[F32]>:$result);

    // Define how to build the operation
    let builders = [
        OpBuilder<(ins "Value":$input_tensor, "Value":$weight_tensor, "Value":$bias_tensor, "ArrayAttr":$dilation, "ArrayAttr":$padding, "ArrayAttr":$stride, "FloatAttr":$max_fp, "FloatAttr":$min_fp)>
    ];
}

#endif // _COMPILER_INCLUDES_DIALECT_TINYFUSIONOPS_TD_
