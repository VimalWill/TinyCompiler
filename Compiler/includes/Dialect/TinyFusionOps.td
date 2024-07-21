#ifndef _COMPILER_INCLUDES_DIALECT_TINYFUSIONOPS_TD_
#define _COMPILER_INCLUDES_DIALECT_TINYFUSIONOPS_TD_

include "Dialect/TinyFusionDialect.td"

class Tiny_Fusion_Op<string mnemonic, list<Trait> traits = []> : Op<Tiny_Fusion_Dialect, mnemonic, traits>;

def Conv2DElemOp : Tiny_Fusion_Op<"con2d_elemwise", [NoSideEffect]> {
    let summary = "conv2d + elemwise operator"; 
    let description [{
        fuse conv2d operator with the following 
        element wise operator 

        ```
        TinyFusion.conv2d_elemwise
        ```
    }]; 

    //ref: https://chatgpt.com/c/18a96241-6e4a-439a-b494-4208436d6598
}

#endif //_COMPILER_INCLUDES_DIALECT_TINYFUSIONOPS_TD_