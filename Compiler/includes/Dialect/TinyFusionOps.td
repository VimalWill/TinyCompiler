#ifndef _COMPILER_INCLUDES_DIALECT_TINYFUSIONOPS_TD_
#define _COMPILER_INCLUDES_DIALECT_TINYFUSIONOPS_TD_

include "Dialect/TinyFusionDialect.td"


class Tiny_Fusion_Op<string mnemonic, list<Trait> traits = []> : Op<Tiny_Fusion_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Conv2dReLuOp
//===----------------------------------------------------------------------===//
def Conv2dReluOp : Tiny_Fusion_Op<"conv2d_relu", [NoSideEffect]> {
    let summary = "conv2d + relu fusion operation"; 
    let description = [{
        Fuse complex operator (conv2d) with the elemwise 
        operator (relu)
        ```
            TinyFusion.conv2d_relu
        ```
    }];

    let arguments = (ins 
        TensorOf<[F32]>:$input_tensor, 
        TensorOf<[F32]>:$weight_tensor, 
        TensorOf<[F32]>:$bias_tensor, 
        ArrayAttr<DilationAttr>:$dilation, 
        ArrayAttr<PaddingAttr>:$padding, 
        ArrayAttr<StrideAttr>:$stride,
        FloatAttr:$max_fp,  
        FloatAttr:$min_fp  
    );

    let assemblyFormat = [{
        $input_tensor `,` $weight_tensor `,` $bias_tensor ` ` attr-dict ` : ` type($input_tensor) `,` type($weight_tensor) `,` type($bias_tensor) ` -> ` type($result)
    }];

    let builders = [
        OpBuilder<(ins "Value":$input_tensor, "Value":$weight_tensor, "Value":$bias_tensor, "ArrayAttr":$dilation, "ArrayAttr":$padding, "ArrayAttr":$stride, "FloatAttr":$min_fp, "FloatAttr":$max_fp)>
    ];
}

#endif // _COMPILER_INCLUDES_DIALECT_TINYFUSIONOPS_TD_
