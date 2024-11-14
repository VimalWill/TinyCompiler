#include "Dialect/Passes.h"
#include "Dialect/TinyFusionDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Conversion/TosaToTensor/TosaToTensor.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

using namespace mlir;
using namespace mlir::TinyFusion;

void TinyCompilerPipeline(mlir::OpPassManager &pm) {
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::TinyFusion::registerLowerToTinyFusionPass());
  // pm.addNestedPass<mlir::func::FuncOp>(mlir::tosa::createTosaToTensor());
  // pm.addPass(mlir::bufferization::createEmptyTensorToAllocTensorPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::TinyFusion::registerLowerToAffinePass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createSCCPPass());
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    llvm::errs() << "Expected at least one input file in .mlir format\n";
    return 1;
  }

  mlir::registerAllPasses();
  // mlir::TinyFusion::registerLowerToTinyFusionPass();
  // mlir::TinyFusion::registerLowerToAffinePass();
  mlir::DialectRegistry registry;
  registry.insert<mlir::tosa::TosaDialect, mlir::TinyFusion::TinyFusionDialect,
                  mlir::func::FuncDialect, mlir::affine::AffineDialect,
                  mlir::arith::ArithDialect>();

  mlir::MLIRContext context(registry);

  mlir::PassPipelineRegistration<> pipeline(
      "compile", "Runs lowering from TOSA to Arith, including fusion",
      TinyCompilerPipeline);

  return asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "TinyCompiler-opt", registry));
}
