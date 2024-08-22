#include "Dialect/TinyFusionDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::tosa;
using namespace mlir::func;
using namespace mlir::affine;
using namespace mlir::TinyFusion;

namespace {

// ref:
// https://www.lei.chat/posts/mlir-codegen-dialects-for-machine-learning-compilers/

// dialect lowering from TinyFusion.conv2d_relu to affine.for
// ref: https://mlir.llvm.org/docs/Tutorials/Toy/Ch-5/
struct Conv2dReluOpLowering : public mlir::ConversionPattern {
  Conv2dReluOpLowering(mlir::MLIRContext *ctx)
      : mlir::ConversionPattern(Conv2dReluOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *Op, ArrayRef<Value> Operands,
                                ConversionPatternRewriter &rewriter) override { /*todo: affine logic*/ }
};
} // namespace

namespace {

struct LowerTinyFusionToAffine
    : public PassWrapper<LowerTinyFusionToAffine, OperationPass<func::FuncOp>> {

public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerTinyFusionToAffine);

  StringRef getArgument() const final { return "lower-tinyfusion-to-affine"; }
  StringRef getDescription() const final {
    return "lower TinyFusion to Affine dialect";
  }

  void getDependentDialect(DialectRegistry &registry) const override {
    registry.insert<TinyFusion::TinyFusionDialect, tosa::TosaDialect,
                    func::FuncDialect, affine::AffineDialect>();
  }

  void runOnOperation() override {
    auto func = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
  }
};
} // namespace