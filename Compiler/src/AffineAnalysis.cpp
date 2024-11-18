#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "Dialect/Passes.h"

using namespace mlir;


int64_t getNestLoop(mlir::Operation *op, int64_t depth = 0) {
    if (auto forOp = dyn_cast<affine::AffineForOp>(op)) {
        for (auto &nestedOp : forOp.getBody()->getOperations()) {
            return getNestLoop(&nestedOp, depth + 1);
        }
    }
    return depth;
}

namespace {
struct TileFessibilityAnalysis
    : public PassWrapper<TileFessibilityAnalysis, OperationPass<func::FuncOp>> {

public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TileFessibilityAnalysis);

  StringRef getArgument() const final { return "tile-fessibility-analysis"; }
  StringRef getDescription() const final {
    return "Analyze the feasibility of tiling";
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    funcOp.walk([&](mlir::Operation *op) {
        int64_t count = getNestLoop(op); 
        llvm::errs() << count << "\n"; 
    });
  }
};
} // namespace

std::unique_ptr<mlir::Pass> mlir::TinyFusion::registerLoopAnalysisPass() {
  return std::make_unique<TileFessibilityAnalysis>();
}
