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

using namespace mlir;
using namespace mlir::tosa;
using namespace mlir::func;
using namespace mlir::arith;
using namespace mlir::memref;
using namespace mlir::affine;
using namespace mlir::TinyFusion;

namespace {

// __attribute__((__always_inline__)) static MemRefType
// convertTensorToMemref(auto rankedTenorType) {
//   return MemRefType::get(rankedTensorType.getShape(),
//                          rankedTensorType.getElementType());
// }

// ref:
// https://blog.weghos.com/llvm/llvm/mlir/examples/toy/Ch5/mlir/LowerToAffineLoops.cpp.html
// struct Conv2dReluOpLowering : public mlir::ConversionPattern {
//   Conv2dReluOpLowering(mlir::MLIRContext *ctx)
//       : mlir::ConversionPattern(Conv2dReluOp::getOperationName(), 1, ctx) {}

//   LogicalResult matchAndRewrite(Operation *Op, ArrayRef<Value> Operands,
//                                 ConversionPatternRewriter &rewriter) override
//                                 {}
// };

struct ConstantOpLowering : public OpConversionPattern<tosa::ConstOp> {
  ConstantOpLowering(mlir::MLIRContext *ctx) : OpConversionPattern(ctx) {}
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tosa::ConstOp constOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ImplicitLocOpBuilder b(constOp.getLoc(), rewriter);
    auto arithConstOp = b.create<arith::ConstantOp>(adaptor.getValue());
    rewriter.replaceOp(constOp, arithConstOp);
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
                    arith::ArithDialect>();
  }

  void runOnOperation() override {
    auto func = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    patterns.add<ConstantOpLowering>(context);
    ConversionTarget target(*context);
    target.addIllegalOp<tosa::ConstOp>();

    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

void mlir::TinyFusion::registerLowerToAffinePass() {
  PassRegistration<LowerTinyFusionToAffine>();
}