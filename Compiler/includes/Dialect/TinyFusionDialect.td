// --- TinyFusionDialect.td - TinyFusionDialect --- 

#ifndef _COMPILER_INCLUDES_TINYFUSIONDIALECT_TD_
#define _COMPILER_INCLUDES_TINYFUSIONDIALECT_TD_

include "mlir/IR/OpBase.td"
include "mlir/IR/DialectBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Pass/PassBase.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// Tiny Fusion Dialect
//===----------------------------------------------------------------------===//
def Tiny_Fusion_Dialect : Dialect {

    let name = "TinyFusion"; 
    let summary = "tiny-fusion dialect for operator fusion"; 
    let description = [{
        fusion dialect of tiny compiler applies hardware-indep
        operator fusion to the DNN operators. The dialect can 
        be the lower level abstruction to TOSA dialect.
        
        ref paper: https://layman-n-ish.github.io/pdfs/TVM_Review_Report.pdf
    }]; 

    let cppNamespace = "::mlir::tinyfusion"; 
}

#endif //_COMPILER_INCLUDES_TINYFUSIONDIALECT_TD_