#ifndef _COMPILER_INCLUDES_DIALECT_PASSES_H_
#define _COMPILER_INCLUDES_DIALECT_PASSES_H_

#include <memory>

namespace mlir {
    class Pass; 

    namespace TinyFusion {
        std::unique_ptr<mlir::Pass> createComplexFuseRelu(); 
    }
}

#endif //_COMPILER_INCLUDES_DIALECT_PASSES_H_