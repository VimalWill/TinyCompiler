#include "Dialect/TinyFusionDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"

#include "Dialect/Passes.h"
#include "Dialect/TinyFusionDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/Sequence.h"
#include <memory>

using namespace mlir;
using namespace mlir::tosa;
using namespace mlir::func;
using namespace mlir::arith;
using namespace mlir::memref;
using namespace mlir::affine;
using namespace mlir::TinyFusion;

namespace {

// ref:
// https://blog.weghos.com/llvm/llvm/mlir/examples/toy/Ch5/mlir/LowerToAffineLoops.cpp.html
// struct Conv2dReluOpLowering : public mlir::ConversionPattern {
//   Conv2dReluOpLowering(mlir::MLIRContext *ctx)
//       : mlir::ConversionPattern(Conv2dReluOp::getOperationName(), 1, ctx) {}

//   LogicalResult matchAndRewrite(Operation *Op, ArrayRef<Value> Operands,
//                                 ConversionPatternRewriter &rewriter) override
//                                 {}
// };

struct PadOpLowering : public OpRewritePattern<tosa::ReshapeOp> {
  using OpRewritePattern<tosa::ReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::ReshapeOp reshapeOp,
                                PatternRewriter &rewriter) const override {
    auto op = reshapeOp->getNextNode();
    if (!op)
      return failure();

    auto paddingAttr = op->getAttrOfType<ArrayAttr>("padding");
    if (paddingAttr.getValue().empty())
      return failure();
    
    return success(); 
  }
};

struct ConstantOpLowering : public OpRewritePattern<tosa::ConstOp> {
  using OpRewritePattern<tosa::ConstOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::ConstOp constOp,
                                PatternRewriter &rewriter) const override {

    auto constantValue = constOp.getValue();
    if (!constantValue)
      return failure();

    auto Loc = constOp.getLoc();
    auto arithConstantOp = rewriter.create<arith::ConstantOp>(
        Loc, constOp.getType(), constantValue);

    rewriter.replaceOp(constOp, arithConstantOp.getResult());
    return success();
  }
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

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TinyFusion::TinyFusionDialect, tosa::TosaDialect,
                    func::FuncDialect, affine::AffineDialect,
                    arith::ArithDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    auto func = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    patterns.add<ConstantOpLowering>(context);
    patterns.add<PadOpLowering>(context);
    ConversionTarget target(*context);
    target.addIllegalOp<tosa::ConstOp>();

    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass> mlir::TinyFusion::registerLowerToAffinePass() {
  return std::make_unique<LowerTinyFusionToAffine>();
}