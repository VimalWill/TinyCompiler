#include "Dialect/TinyFusionDialect.h"
#include "Dialect/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::tosa;
using namespace mlir::func;
using namespace mlir::TinyFusion;

namespace {
    class LowerToConv2dRelu : public OpRewritePattern<tosa::Conv2DOp> {
    public:
        using OpRewritePattern<tosa::Conv2DOp>::OpRewritePattern;

        LogicalResult matchAndRewrite(tosa::Conv2DOp convOp, PatternRewriter &rewriter) const override {
            // Check if the next operations are Reshape and Clamp
            auto reshapeOp = dyn_cast_or_null<tosa::ReshapeOp>(convOp->getNextNode());
            if (!reshapeOp) return failure();

            auto clampOp = dyn_cast_or_null<tosa::ClampOp>(reshapeOp->getNextNode());
            if (!clampOp) return failure();

            auto input = convOp.getOperand(0); 
            auto filter = convOp.getOperand(1); 
            auto bias = convOp.getOperand(2); 

            auto dilation = rewriter.getI64ArrayAttr(convOp.getDilation());
            auto padding = rewriter.getI64ArrayAttr(convOp.getPad());
            auto stride = rewriter.getI64ArrayAttr(convOp.getStride());

            auto max_fp = rewriter.getF32FloatAttr(clampOp.getMaxFp().convertToFloat());
            auto min_fp = rewriter.getF32FloatAttr(clampOp.getMinFp().convertToFloat());

            rewriter.replaceOpWithNewOp<Conv2dReluOp>(
                clampOp, reshapeOp.getType(), input, filter, bias, dilation, padding, stride, max_fp, min_fp
            );

            rewriter.eraseOp(reshapeOp);
            return success();
        }
    };
}

namespace {
    struct LowerTosaToTinyFusion : public PassWrapper<LowerTosaToTinyFusion, OperationPass<func::FuncOp>> {
        // MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerTosaToTinyFusion);

        void runOnOperation() override {
            auto func = getOperation();
            MLIRContext *context = &getContext();
            RewritePatternSet patterns(context);

            patterns.add<LowerToConv2dRelu>(context);

            if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
                signalPassFailure();
            }
        }
    };
}

std::unique_ptr<mlir::Pass> mlir::TinyFusion::createLowerToTinyFusionPass() {
    return std::make_unique<LowerTosaToTinyFusion>();
}
