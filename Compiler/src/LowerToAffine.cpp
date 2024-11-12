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
#include <vector>

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

struct AffineOpLowering : public OpRewritePattern<TinyFusion::Conv2dReluOp> {
  using OpRewritePattern<TinyFusion::Conv2dReluOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TinyFusion::Conv2dReluOp conv2dReluOp,
                                PatternRewriter &rewriter) const override {
    auto loc = conv2dReluOp.getLoc();
    auto paddingAttr = conv2dReluOp.getPaddingAttr();
    auto padType = RankedTensorType::get({4, 2}, rewriter.getI64Type());

    llvm::SmallVector<int64_t, 8> paddingVal = {0, 0, 0, 0, 0, 0, 0, 0};
    unsigned i = 4;
    for (auto val : paddingAttr) {
      if (auto tmp = dyn_cast<IntegerAttr>(val)) {
        paddingVal[i++] = tmp.getInt();
      }
    }

    auto padConstAttr = DenseIntElementsAttr::get(padType, paddingVal);
    auto padDimConstOp = rewriter.create<arith::ConstantOp>(loc, padConstAttr);

    auto reshapeOp =
        conv2dReluOp.getOperands()[0].getDefiningOp<tosa::ReshapeOp>();
    if (!reshapeOp)
      return failure();

    auto inputTensor = reshapeOp.getResult();
    int64_t cc = inputTensor.getType().getShape()[0];
    int64_t cb = inputTensor.getType().getShape()[3];
    int64_t ph = inputTensor.getType().getShape()[1] + paddingVal[4];
    int64_t pw = inputTensor.getType().getShape()[2] + paddingVal[6];
    auto padOutputType =
        RankedTensorType::get({cc, ph, pw, cb}, rewriter.getF32Type());
    auto tosaPadOp = rewriter.create<tosa::PadOp>(loc, padOutputType,
                                                  inputTensor, padDimConstOp);
    if (!tosaPadOp)
      return failure();

    // TODO: lower TinyFusion.conv2d_relu dialect to affine.for
    // https://discourse.llvm.org/t/mlir-lowering-customop-to-affine-nested-for/83083

    auto weightType = conv2dReluOp.getOperands()[1].getType().cast<ShapedType>();
    int64_t kh = weightType.getShape()[1]; 
    int64_t kw = weightType.getShape()[2]; 

    auto outputTensor = conv2dReluOp.getResult().getType().cast<ShapedType>(); 
    int64_t o_cb = outputTensor.getShape()[0]; 
    int64_t o_cc = outputTensor.getShape()[3]; 
    int64_t o_ch = outputTensor.getShape()[1]; 
    int64_t o_cw = outputTensor.getShape()[2]; 

    //todo: refactor here
    auto outputBatchLoop = rewriter.create<affine::AffineForOp>(loc, 0, o_cb, 1);
    rewriter.setInsertionPointToStart(outputBatchLoop.getBody());

    auto outputChannelLoop = rewriter.create<affine::AffineForOp>(loc, 0, o_cc, 1);
    rewriter.setInsertionPointToStart(outputChannelLoop.getBody());

    auto outputHeightLoop = rewriter.create<affine::AffineForOp>(loc, 0, o_ch, 1);
    rewriter.setInsertionPointToStart(outputHeightLoop.getBody());
    auto outputWidthLoop = rewriter.create<affine::AffineForOp>(loc, 0, o_cw, 1);
    rewriter.setInsertionPointToStart(outputWidthLoop.getBody());

    auto kernelHeightLoop = rewriter.create<affine::AffineForOp>(loc, 0, kh, 1); 
    rewriter.setInsertionPointToStart(kernelHeightLoop.getBody());
    auto kernelWidthLoop = rewriter.create<affine::AffineForOp>(loc, 0, kw, 1); 
    rewriter.setInsertionPointToStart(kernelWidthLoop.getBody());


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
    patterns.add<AffineOpLowering>(context);
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
