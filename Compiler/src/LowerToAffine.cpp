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

void convertToAffineFor(tosa::ReshapeOp op) {

  auto tempOp = op->getNextNode();
  if (!tempNode)
    return;

  if (auto convOp = dyn_cast_or_null<TinyFusion::Conv2dReluOp>(tempOp)) {

    auto inputType = convOp.getOperands()[0].getType().cast<ShapedType>();
    auto filterType = convOp.getOperands()[1].getType().cast<ShapedType>();
    auto outputType = convOp.getResult().getType().cast<ShapedType>();

    const int kernelWidth = filterType.getShape()[1];
    const int kernelHeight = filterType.getShape()[2];
    assert(kernelHeight == kernelWidth);

    const int inputWidth = inputType.getShape()[3];
    const int inputHeight = inputType.getShape()[2];
    const int outputWidth = outputType.getShape()[3]; 
    const int outputHeight = outputType.getShape()[2]; 

    /*todo: get dilation, stride and padding*/
  }
}
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

    SmallVector<int64_t, 4> paddingValue;
    for (int itr = 0; itr < 4; itr++) {
      if (auto intAttr = dyn_cast<IntegerAttr>(paddingAttr[itr])) {
        paddingValue.push_back(intAttr.getInt());
      } else {
        return failure();
      }
    }

    auto paddingType =
        RankedTensorType::get({4, 2}, rewriter.getIntegerType(64));
    auto paddingDenseAttr =
        DenseElementsAttr::get(paddingType, llvm::ArrayRef(paddingValue));
    auto padConstOp = rewriter.create<arith::ConstantOp>(
        op->getLoc(), paddingType, paddingDenseAttr);

    auto padOp = rewriter.create<tosa::PadOp>(
        op->getLoc(), op->getResult(0).getType(), op->getResult(0), padConstOp);

    convertToAffineFor(reshapeOp);

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