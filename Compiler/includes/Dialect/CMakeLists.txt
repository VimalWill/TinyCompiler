set(CMAKE_CURRENT_BINARY_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(dialect TinyFusion)
set(dialect_namespace TinyFusion)
set(LLVM_TARGET_DEFINITIONS ${dialect}Dialect.td)

mlir_tablegen(${dialect}Ops.h.inc -gen-op-decls)
mlir_tablegen(${dialect}Ops.cpp.inc -gen-op-defs)
mlir_tablegen(${dialect}Dialect.h.inc -gen-dialect-decls
              -dialect=${dialect_namespace})
mlir_tablegen(${dialect}Dialect.cpp.inc -gen-dialect-defs
              -dialect=${dialect_namespace})


add_public_tablegen_target(MLIR${dialect}IncGen)