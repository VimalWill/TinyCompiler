/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: TinyFusionOps.td                                                     *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace TinyFusion {

class TinyFusionDialect : public ::mlir::Dialect {
  explicit TinyFusionDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~TinyFusionDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("TinyFusion");
  }
};
} // namespace TinyFusion
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::TinyFusion::TinyFusionDialect)
