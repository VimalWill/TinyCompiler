#include "Dialect/TinyFusionDialect.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
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

int main(int argc, char** argv) {
    mlir::DialectRegistry registry; 
    registry.insert<mlir::tosa::TosaDialect, mlir::TinyFusion::TinyFusionDialect>(); 
    registerAllDialects(registry);    

    return 0; 
}
