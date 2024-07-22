#include "includes/Dialect/TinyFusionDialect.h"
#include "includes/Dialect/Passes.h"

using mlir; 
using mlir::TinyFusion; 

namespace {

    struct LowerTosaToTinyFusionOp : PassWrapper<LowerTosaToTinyFusionOp,/*Todo*/> {

        //ref: https://chatgpt.com/c/0b521c1f-afce-417b-81c8-585203247604

    }; 
} //namespace