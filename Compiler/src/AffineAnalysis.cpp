#include "Dialect/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#define CONV_TILE_DEPTH 7

using namespace mlir;

int64_t getNestDepth(mlir::Operation *op, int64_t depth = 0) {
  if (auto forOp = dyn_cast<affine::AffineForOp>(op)) {
    for (auto &nestedOp : forOp.getBody()->getOperations()) {
      return getNestDepth(&nestedOp, depth + 1);
    }
  }
  return depth;
}

void estimateTileSize(mlir::Operation *op) {
  auto forOp = dyn_cast<affine::AffineForOp>(op);
  auto lb = forOp.getLowerBoundMap();
  auto ub = forOp.getUpperBoundMap();

  if (!lb || !ub)
    return;
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
    func::FuncOp func = getOperation();
    func.walk([&](Operation *op) {
      int64_t depth = 0;
      if ((depth = getNestDepth(op)) == CONV_TILE_DEPTH) {
        estimateTileSize(op);
      }
    });
  }
};
} // namespace

std::unique_ptr<mlir::Pass> mlir::TinyFusion::registerLoopAnalysisPass() {
  return std::make_unique<TileFessibilityAnalysis>();
}
