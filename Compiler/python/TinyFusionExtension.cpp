#include "Dialect/TinyFusionCApi.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

using namespace py::literals;
using namespace mlir::python::adaptors;

PYBIND11_MODULE(_TinyFusion, m) {
  m.doc() = "mlir-TinyFusion main python extern";


  m.def(
      "register_TinyFusion_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle TinyFusionDialect = mlirGetDialectHandle__TinyFusion__();
        mlirDialectHandleRegisterDialect(TinyFusionDialect, context);
        if (load) {
          mlirDialectHandleLoadDialect(TinyFusionDialect, context);
        }
      },
      py::arg("context"), py::arg("load") = true);
}