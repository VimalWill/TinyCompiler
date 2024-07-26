#include "Dialect/TinyFusionCApi.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

using namespace py::literals;
using namespace mlir::python::adaptors;

PYBIND11_MODULE(_minimal, m) {
  //===--------------------------------------------------------------------===//
  // minimal dialect
  //===--------------------------------------------------------------------===//
  m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle handle = mlirGetDialectHandle__TinyFusion__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load) {
          mlirDialectHandleLoadDialect(handle, context);
        }
      },
      "context"_a = py::none(), "load"_a = true);
}