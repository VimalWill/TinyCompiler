#ifndef _COMPILER_INCLUDES_DIALECT_PASSES_H_
#define _COMPILER_INCLUDES_DIALECT_PASSES_H_

#include <memory>

namespace mlir {
class Pass;

namespace TinyFusion {
// std::unique_ptr<mlir::Pass> createLowerToTinyFusionPass();
std::unique_ptr<mlir::Pass> registerLowerToTinyFusionPass();
std::unique_ptr<mlir::Pass> registerLowerToAffinePass();
std::unique_ptr<mlir::Pass> registerLoopAnalysisPass(); 
} // namespace TinyFusion
} // namespace mlir

#endif //_COMPILER_INCLUDES_DIALECT_PASSES_H_