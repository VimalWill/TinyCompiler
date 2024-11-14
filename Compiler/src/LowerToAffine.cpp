#include "Dialect/TinyFusionDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"

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
using namespace mlir::tensor;
using namespace mlir::TinyFusion;

mlir::affine::AffineForOp Loop(PatternRewriter &rewriter, Location &loc,
                               int64_t lower, int64_t upper) {
  auto loop = rewriter.create<affine::AffineForOp>(loc, lower, upper, 1);
  rewriter.setInsertionPointToStart(loop.getBody());
  return loop;
}

static MemRefType convertTensorToMemRef(RankedTensorType type) {
  return MemRefType::get(type.getShape(), type.getElementType());
}

static Value insertAllocAndDealloc(MemRefType type, Location loc,
                                   PatternRewriter &rewriter) {
  auto alloc = rewriter.create<memref::AllocOp>(loc, type);

  auto *parentBlock = alloc->getBlock();
  alloc->moveBefore(&parentBlock->front());

  auto dealloc = rewriter.create<memref::DeallocOp>(loc, alloc);
  dealloc->moveBefore(&parentBlock->back());
  return alloc;
}

namespace {

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

    auto originalInsertionPoint = rewriter.saveInsertionPoint();
    auto weightConstop =
        conv2dReluOp.getOperands()[1].getDefiningOp<arith::ConstantOp>();
    if (!weightConstop)
      return failure();
    auto biasConstop =
        conv2dReluOp.getOperands()[2].getDefiningOp<arith::ConstantOp>();
    if (!biasConstop)
      return failure();

    auto weightType =
        conv2dReluOp.getOperands()[1].getType().cast<RankedTensorType>();
    int64_t kh = weightType.getShape()[1];
    int64_t kw = weightType.getShape()[2];
    int64_t wb = weightType.getShape()[0];
    int64_t wc = weightType.getShape()[3];

    MemRefType weightMemRefType = convertTensorToMemRef(weightType);
    Value weightMemRef = insertAllocAndDealloc(weightMemRefType, loc, rewriter);

    auto wB = Loop(rewriter, loc, 0, wb);
    auto wH = Loop(rewriter, loc, 0, kh);
    auto wW = Loop(rewriter, loc, 0, kw);
    auto wC = Loop(rewriter, loc, 0, wc);

    auto b = wB.getInductionVar();
    auto h = wH.getInductionVar();
    auto w = wW.getInductionVar();
    auto c = wC.getInductionVar();

    auto weightTmp = rewriter.create<tensor::ExtractOp>(loc, weightConstop,
                                                        ValueRange{b, h, w, c});
    rewriter.create<memref::StoreOp>(loc, weightTmp, weightMemRef,
                                     ValueRange{b, h, w, c});

    rewriter.restoreInsertionPoint(originalInsertionPoint);
    RankedTensorType biasTensorType =
        conv2dReluOp.getOperands()[2].getType().cast<RankedTensorType>();
    MemRefType biasMemRefType = convertTensorToMemRef(biasTensorType);
    Value biasMemRef = insertAllocAndDealloc(biasMemRefType, loc, rewriter);

    int64_t bs = biasTensorType.getShape()[0];
    auto bW = Loop(rewriter, loc, 0, bs);

    auto s = bW.getInductionVar();
    auto biasTmp =
        rewriter.create<tensor::ExtractOp>(loc, biasConstop, ValueRange{s});
    rewriter.create<memref::StoreOp>(loc, biasTmp, biasMemRef, ValueRange{s});
    rewriter.restoreInsertionPoint(originalInsertionPoint);

    MemRefType padOutputMemRefType = convertTensorToMemRef(padOutputType);
    auto actMemRef = rewriter.create<bufferization::ToMemrefOp>(loc, padOutputMemRefType,tosaPadOp.getResult());
    if(!actMemRef)
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
                    arith::ArithDialect, memref::MemRefDialect, bufferization::BufferizationDialect>();
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
