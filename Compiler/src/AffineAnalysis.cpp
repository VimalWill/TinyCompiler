#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace {

struct TileFessibilityPass
    : public PassWrapper<TileFessibilityPass,
                         operationPass<mlir::func::FuncOp>> {
    
    void runOnOperation() {
        auto func = getOperation();
    }                   
};
} // namespace