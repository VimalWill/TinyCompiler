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
#include "mlir/InitAllPasses.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;
using namespace mlir::TinyFusion;


// void TinyFusionPipeline(OpPassManager& manager) {
//     manager.addNestedPass<func::FuncOp>(LowerTosaToTinyFusion());
//     manager.addPass(createCanonicalizerPass());
//     manager.addPass(createCSEPass());
// }

int main(int argc, char** argv) {
    mlir::registerAllPasses(); 
    mlir::TinyFusion::registerLowerToTinyFusionPass(); 
    mlir::DialectRegistry registry;
    registry.insert<mlir::tosa::TosaDialect, mlir::TinyFusion::TinyFusionDialect, mlir::func::FuncDialect>();

    mlir::MLIRContext context(registry);

    // mlir::PassPipelineRegistration<>("test-tosa-to-tinyfusion",
    //     "Run passes to lower TOSA models to TinyFusion",
    //     TinyFusionPipeline);

    // Add debugging output to verify dialect loading
    llvm::errs() << "Registered Dialects:\n";
    for (auto dialect : registry.getDialectNames()) {
        llvm::errs() << "Registered Dialect: " << dialect  << "\n";
    }


    return asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "TinyCompiler-opt", registry));

}

