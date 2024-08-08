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

    let results = (outs TensorOf<[F32]>:$result);
    let builders = [
        OpBuilder<(ins "Value":$input_tensor, "Value":$weight_tensor, "Value":$bias_tensor, "ArrayAttr":$dilation, "ArrayAttr":$padding, "ArrayAttr":$stride, "FloatAttr":$max_fp, "FloatAttr":$min_fp)>
    ];

    let assemblyFormat = [{
        $input_tensor `,`
        $weight_tensor `,`
        $bias_tensor attr-dict `:` `(` type($input_tensor) `,`  type($weight_tensor) `,` type($bias_tensor) `)` `->` type(results)
    }]; 
}


//===----------------------------------------------------------------------===//
// Conv2dSiluOp
//===----------------------------------------------------------------------===//
def Conv2dSiluOp : Tiny_Fusion_Op<"conv2d_silu", [Pure]> {
    let summary = "Conv2d (complex op) can be fused with the Silu (elemwise) followed by"; 
    let description = [{ TinyFusion.conv2d_silu }]; 

    let arguments = (ins
        TensorOf<[F32]>:$input_tensor,
        TensorOf<[F32]>:$weight_tensor,
        TensorOf<[F32]>:$bias_tensor, 
        ArrayAttr:$dilation, 
        ArrayAttr:$padding,
        ArrayAttr:$stride
    ); 

    let results = (outs TensorOf<[F32]>:$result); 
    let builders = [
        OpBuilder<(ins "Value":$input_tensor, "Value":$weight_tensor, "Value":$bias_tensor, "ArrayAttr":$dilation, "ArrayAttr":$padding, "ArrayAttr":$stride)>
    ];

    let assemblyFormat = [{
        $input_tensor `,` 
        $weight_tensor `,` 
        $bias_tensor attr-dict `:` `(` type($input_tensor) `,` type($weight_tensor) `,` type($bias_tensor) `)` `->` type(results)
    }];
}

//===----------------------------------------------------------------------===//
// Conv2dLReluOp
//===----------------------------------------------------------------------===//
def Conv2dLReluOp : Tiny_Fusion_Op<"conv2d_lrelu", [Pure]> {
    let summary = "conv2d (complex op) can be fused with the leaky_relu (elemwise) followed by"; 
    let description = [{ TinyFusion.conv2d_lrelu }]; 

    let arguments = (ins
        TensorOf<[F32]>:$input_tensor, 
        TensorOf<[F32]>:$weight_tensor,
        TensorOf<[F32]>:$bias_tensor, 
        TensorOf<[F32]>:$neg_slop,
        TensorOf<[F32]>:$scale_const,
        ArrayAttr:$dilation, 
        ArrayAttr:$padding, 
        ArrayAttr:$stride
    ); 


    let results = (outs TensorOf<[F32]>:$result); 
    let builders = [
        OpBuilder<(ins "Value":$input_tensor, "Value":$weight_tensor, "Value":$bias_tensor, "Value":$neg_slope, "Value":$scale_const, "ArrayAttr":$dilation, "ArrayAttr":$padding, "ArrayAttr":$stride)>
    ]; 

    let assemblyFormat = [{
        $input_tensor `,` 
        $weight_tensor `,`
        $bias_tensor `,`
        $neg_slop `,`
        $scale_const attr-dict `:` `(` type($input_tensor) `,` type($weight_tensor) `,` type($bias_tensor) `,` type($neg_slop) `,` type($scale_const)`)` `->` type(results)
    }];
}
#endif // _COMPILER_INCLUDES_DIALECT_TINYFUSIONOPS_TD_
