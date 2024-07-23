add_mlir_dialect_library(
  MLIRTinyFusion
  TinyFusionDialect.cpp
  LowerToTinyFusion.cpp

  # to silence LLVMProcessSources.cmake:116
  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/includes

  DEPENDS
  MLIRTinyFusionIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRFuncDialect
)