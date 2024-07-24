#include "Dialect/TinyFusionDialect.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "Dialect/Passes.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"

#include "mlir/InitAllDialects.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

using namespace mlir;
using namespace mlir::TinyFusion;

// Custom pass pipeline for TinyFusion
void TinyFusionPipeline(mlir::OpPassManager& manager) {
    manager.addNestedPass<func::FuncOp>(mlir::TinyFusion::createLowerToTinyFusionPass());
    manager.addPass(mlir::createCanonicalizerPass());
    manager.addPass(mlir::createCSEPass());
}

int main(int argc, char** argv) {
    // Register TOSA and TinyFusion dialects
    // mlir::DialectRegistry registry;
    // registry.insert<mlir::tosa::TosaDialect, mlir::TinyFusion::TinyFusionDialect>();
    // registerAllDialects(registry);

    // // Create MLIR context and load all available dialects
    // mlir::MLIRContext context(registry);
    // context.loadAllAvailableDialects();
    MLIRContext ctx;
    ctx.loadDialect<mlir::func::FuncDialect, mlir::arith::ArithDialect, mlir::tosa::TosaDialect, mlir::TinyFusion::TinyFusionDialect>();

    auto src = parseSourceFile<ModuleOp>(argv[1], &ctx);
    src->print(llvm::outs());

    return 0; 
}
