/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: TinyFusionOps.td                                                     *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::TinyFusion::TinyFusionDialect)
namespace mlir {
namespace TinyFusion {

TinyFusionDialect::TinyFusionDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<TinyFusionDialect>())
    
     {
  getContext()->loadDialect<tosa::TosaDialect>();
  initialize();
}

TinyFusionDialect::~TinyFusionDialect() = default;

} // namespace TinyFusion
} // namespace mlir
