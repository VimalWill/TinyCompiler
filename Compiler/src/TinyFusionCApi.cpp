#include "Dialect/TinyFusionCApi.h"
#include "Dialect/TinyFusionDialect.h"
#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(TinyFusion, TinyFusion, mlir::TinyFusion::TinyFusionDialect)