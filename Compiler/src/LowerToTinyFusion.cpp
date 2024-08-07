#include "Dialect/Passes.h"
#include "Dialect/TinyFusionDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <iostream>

using namespace mlir;
using namespace mlir::tosa;
using namespace mlir::func;
using namespace mlir::TinyFusion;

namespace {
class LowerToConv2dSilu : public OpRewritePattern<tosa::Conv2DOp> {
public:
  using OpRewritePattern<tosa::Conv2DOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::Conv2DOp convOp,
                                PatternRewriter &rewriter) const override {
    auto reshapeAfterConvOp =
        dyn_cast_or_null<tosa::ReshapeOp>(convOp->getNextNode());
    if (!reshapeAfterConvOp)
      return failure();

    auto sigmoidOp =
        dyn_cast_or_null<tosa::SigmoidOp>(reshapeAfterConvOp->getNextNode());
    if (!sigmoidOp)
      return failure();

    auto mulOp = dyn_cast_or_null<tosa::MulOp>(sigmoidOp->getNextNode());
    if (!mulOp)
      return failure();

    auto Input = convOp.getOperand(0);
    auto Weight = convOp.getOperand(1);
    auto Bias = convOp.getOperand(2);

    auto Dilation = rewriter.getI64ArrayAttr(convOp.getDilation());
    auto Padding = rewriter.getI64ArrayAttr(convOp.getPad());
    auto Stride = rewriter.getI64ArrayAttr(convOp.getStride());

    auto fuseOp = rewriter.create<TinyFusion::Conv2dSiluOp>(
        sigmoidOp.getLoc(), convOp.getType(), Input, Weight, Bias, Dilation,
        Padding, Stride);

    auto reshapeOp = rewriter.create<tosa::ReshapeOp>(
        mulOp.getLoc(), reshapeAfterConvOp.getResult().getType(),
        fuseOp.getResult(),
        rewriter.getDenseI64ArrayAttr(reshapeAfterConvOp.getResult()
                                          .getType()
                                          .cast<RankedTensorType>()
                                          .getShape()));

    rewriter.replaceOp(sigmoidOp, fuseOp);
    rewriter.replaceOp(mulOp, reshapeOp);

    // rewriter.eraseOp(convOp);
    // rewriter.eraseOp(reshapeAfterConvOp);

    return success();
  }
};



class LowerToConv2dLRelu : public OpRewritePattern<tosa::Conv2DOp> {
public:
  using OpRewritePattern<tosa::Conv2DOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::Conv2DOp convOp, PatternRewriter &rewriter) const override {

    auto reshapeOpAfterConv = dyn_cast_or_null<tosa::ReshapeOp>(convOp->getNextNode()); 
    if(!reshapeOpAfterConv) return failure(); 
    
    auto maxOp = dyn_cast_or_null<tosa::MaximumOp>(reshapeOpAfterConv->getNextNode()); 
    if(!maxOp) return failure(); 

    auto minOp = dyn_cast_or_null<tosa::MinimumOp>(maxOp->getNextNode()); 
    if(!minOp) return failure(); 

    auto mulOp = dyn_cast_or_null<tosa::MulOp>(minOp->getNextNode()); 
    if(!mulOp) return failure(); 

    auto addOp = dyn_cast_or_null<tosa::AddOp>(mulOp->getNextNode()); 
    if(!addOp) return failure(); 

    auto NegSlope = mulOp.getOperand(1); 

    auto Input = convOp.getOperand(0); 
    auto Weight = convOp.getOperand(1); 
    auto Bias = convOp.getOperand(2); 

    auto dilation = rewriter.getI64ArrayAttr(convOp.getDilation()); 
    auto padding = rewriter.getI64ArrayAttr(convOp.getPad()); 
    auto stride = rewriter.getI64ArrayAttr(convOp.getStride()); 

    auto fuseOp = rewriter.create<TinyFusion::Conv2dLReluOp>(
      convOp.getLoc(), convOp.getType(), Input, Weight, Bias, NegSlope, dilation, 
      padding, stride); 
    
    rewriter.replaceOp(convOp, fuseOp.getResult());
    //TODO: replace mul and add with fuseop and reshape

    // rewriter.eraseOp(reshapeOpAfterConv); 
    // rewriter.eraseOp(convOp); 
    // rewriter.eraseOp(maxOp); 
    // rewriter.eraseOp(minOp); 
    // rewriter.eraseOp(addOp); 
    // rewriter.eraseOp(mulOp); 
    // rewriter.eraseOp(constOp); 

    return success();
  }
};

class LowerToConv2dRelu : public OpRewritePattern<tosa::Conv2DOp> {
public:
  using OpRewritePattern<tosa::Conv2DOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::Conv2DOp convOp,
                                PatternRewriter &rewriter) const override {
    auto reshapeOpBeforeConv =
        convOp.getOperand(0).getDefiningOp<tosa::ReshapeOp>();
    if (!reshapeOpBeforeConv)
      return failure();

    auto reshapeOpAfterConv =
        dyn_cast_or_null<tosa::ReshapeOp>(convOp->getNextNode());
    if (!reshapeOpAfterConv)
      return failure();

    auto clampOp =
        dyn_cast_or_null<tosa::ClampOp>(reshapeOpAfterConv->getNextNode());
    if (!clampOp)
      return failure();

    // Create the fused TinyFusion.conv2d_relu operation
    auto input = convOp.getOperand(0);
    auto filter = convOp.getOperand(1);
    auto bias = convOp.getOperand(2);

    auto dilation = rewriter.getI64ArrayAttr(convOp.getDilation());
    auto padding = rewriter.getI64ArrayAttr(convOp.getPad());
    auto stride = rewriter.getI64ArrayAttr(convOp.getStride());

    auto max_fp = rewriter.getF32FloatAttr(clampOp.getMaxFp().convertToFloat());
    auto min_fp = rewriter.getF32FloatAttr(clampOp.getMinFp().convertToFloat());

    auto fusedOp = rewriter.create<TinyFusion::Conv2dReluOp>(
        reshapeOpAfterConv.getLoc(), convOp.getType(), input, filter, bias,
        dilation, padding, stride, max_fp, min_fp);

    auto shapeOp = rewriter.create<tosa::ReshapeOp>(
        clampOp.getLoc(), reshapeOpAfterConv.getResult().getType(),
        fusedOp.getResult(),
        rewriter.getDenseI64ArrayAttr(reshapeOpAfterConv.getResult()
                                          .getType()
                                          .cast<RankedTensorType>()
                                          .getShape()));

    rewriter.replaceOp(reshapeOpAfterConv, fusedOp.getResult());
    rewriter.replaceOp(clampOp, shapeOp.getResult());

    return success();
  }
};
} // namespace

namespace {
struct LowerTosaToTinyFusion
    : public PassWrapper<LowerTosaToTinyFusion, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerTosaToTinyFusion);

  StringRef getArgument() const final { return "lower-tinyfusion"; }
  StringRef getDescription() const final { return "Lower TinyFusion Dialect."; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TinyFusion::TinyFusionDialect, tosa::TosaDialect,
                    func::FuncDialect>();
  }

  void runOnOperation() override {
    auto func = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    patterns.add<LowerToConv2dRelu>(context);
    patterns.add<LowerToConv2dSilu>(context);
    patterns.add<LowerToConv2dLRelu>(context); 

    ConversionTarget target(*context);
    target.addLegalDialect<TinyFusionDialect>();
    target.addIllegalOp<tosa::Conv2DOp>();
    target.addIllegalOp<tosa::ClampOp>();
    target.addIllegalOp<tosa::ReshapeOp>();

    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

void mlir::TinyFusion::registerLowerToTinyFusionPass() {
  PassRegistration<LowerTosaToTinyFusion>();
}
