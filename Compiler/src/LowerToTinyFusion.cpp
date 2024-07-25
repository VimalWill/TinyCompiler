#include "Dialect/TinyFusionDialect.h"
#include "Dialect/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/DialectConversion.h"

#include <iostream>

using namespace mlir;
using namespace mlir::tosa;
using namespace mlir::func;
using namespace mlir::TinyFusion;

//https://github.com/buddy-compiler/buddy-mlir/blob/main/midend/lib/Conversion/LowerDAP/LowerDAPPass.cpp

namespace {
    class LowerToConv2dRelu : public OpRewritePattern<tosa::Conv2DOp> {
    public:
        using OpRewritePattern<tosa::Conv2DOp>::OpRewritePattern;

        LogicalResult matchAndRewrite(tosa::Conv2DOp convOp, PatternRewriter &rewriter) const override {
            auto reshapeOpBeforeConv = convOp.getOperand(0).getDefiningOp<tosa::ReshapeOp>();
            if (!reshapeOpBeforeConv) {
                llvm::errs() << "No preceding reshape op found!\n"; 
                return failure();
            }

            auto reshapeOpAfterConv = dyn_cast_or_null<tosa::ReshapeOp>(convOp->getNextNode());
            if (!reshapeOpAfterConv) {
                llvm::errs() << "No following reshape op found!\n"; 
                return failure();
            }

            auto clampOp = dyn_cast_or_null<tosa::ClampOp>(reshapeOpAfterConv->getNextNode());
            if (!clampOp) {
                llvm::errs() << "No following clamp op found!\n"; 
                return failure();
            }

            // Create the fused TinyFusion.conv2d_relu operation
            auto input = reshapeOpBeforeConv.getOperand();
            auto filter = convOp.getOperand(1);
            auto bias = convOp.getOperand(2);

            auto dilation = rewriter.getI64ArrayAttr(convOp.getDilation());
            auto padding = rewriter.getI64ArrayAttr(convOp.getPad());
            auto stride = rewriter.getI64ArrayAttr(convOp.getStride());

            auto max_fp = rewriter.getF32FloatAttr(clampOp.getMaxFp().convertToFloat());
            auto min_fp = rewriter.getF32FloatAttr(clampOp.getMinFp().convertToFloat());

            // Replace the sequence with the fused operation
            auto fusedOp = rewriter.create<TinyFusion::Conv2dReluOp>(
                clampOp.getLoc(), reshapeOpAfterConv.getType(), input, filter, bias, 
                dilation, padding, stride, max_fp, min_fp);

            rewriter.replaceOp(clampOp, fusedOp.getResult());

            // Erase intermediate ops
            rewriter.eraseOp(reshapeOpAfterConv);
            rewriter.eraseOp(convOp);
            rewriter.eraseOp(reshapeOpBeforeConv);

            return success();
        }
    };
}

namespace {
    struct LowerTosaToTinyFusion : public PassWrapper<LowerTosaToTinyFusion, OperationPass<func::FuncOp>> {
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
}

// std::unique_ptr<mlir::Pass> mlir::TinyFusion::createLowerToTinyFusionPass() {
//     return std::make_unique<LowerTosaToTinyFusion>();
// }

void mlir::TinyFusion::registerLowerToTinyFusionPass() {
    PassRegistration<LowerTosaToTinyFusion>(); 
}