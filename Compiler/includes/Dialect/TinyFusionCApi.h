#ifndef _TINYFUSION_DIALECT_C_API_H_
#define _TINYFUSION_DIALECT_C_API_H_

#include "mlir-c/RegisterEverything.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(TinyFusion, TinyFusion);

#ifdef __cplusplus
}
#endif

#endif  // _TINYFUSION_DIALECT_C_API_H_