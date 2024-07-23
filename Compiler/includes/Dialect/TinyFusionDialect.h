#ifndef _COMPILER_INCLUDES_DIALECT_TINYFUSIONDIALECT_H_
#define _COMPILER_INCLUDES_DIALECT_TINYFUSIONDIALECT_H_

#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/IRMapping.h"

#include "Dialect/TinyFusionDialect.h.inc"

#define GET_OP_CLASSES
#include "Dialect/TinyFusionOps.h.inc"

#endif //_COMPILER_INCLUDES_DIALECT_TINYFUSIONDIALECT_H_