#include "Dialect/TinyFusionDialect.h"

using namespace mlir;
using namespace mlir::TinyFusion;

#include "Dialect/TinyFusionDialect.cpp.inc"

void TinyFusionDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/TinyFusionOps.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "Dialect/TinyFusionOps.cpp.inc"