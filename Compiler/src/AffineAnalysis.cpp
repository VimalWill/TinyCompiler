#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "Dialect/Passes.h"

using namespace mlir;
namespace {
struct TileFessibilityAnalysis
    : public PassWrapper<TileFessibilityAnalysis, OperationPass<func::FuncOp>> {

public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TileFessibilityAnalysis);

  StringRef getArgument() const final { return "tile-fessibility-analysis"; }
  StringRef getDescription() const final {
    return "analyze the fessibitiy of tiling";
  }

  void runOnOperation() override {
    func::FuncOp Op = getOperation(); 
    Op.walk([&](mlir::Operation *op) {
        if(dyn_cast<affine::AffineForOp>(op))
            llvm::errs() << "found" << "\n"; 
        else 
            llvm::errs() << "not found" << "\n"; 
    }); 
  }
};
} // namespace

std::unique_ptr<mlir::Pass> mlir::TinyFusion::registerLoopAnalysisPass() {
  return std::make_unique<TileFessibilityAnalysis>();
}