include(AddMLIRPython)

declare_mlir_python_sources(MLIRTinyFusionPythonSources)
declare_mlir_python_sources(MLIRTinyFusionPythonExtensions)

declare_mlir_python_sources(MLIRHLOPythonSources.Dialects
  ADD_TO_PARENT MLIRHLOPythonSources
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT MLIRTinyFusionPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  TD_FILE dialects/TinyFusion.td
  SOURCES dialects/TinyFusion.py
  DIALECT_NAME TinyFusion)

declare_mlir_python_extension(MLIRTinyFusionPythonExtensions.Main
  MODULE_NAME _TinyFusion
  ADD_TO_PARENT MLIRTinyFusionPythonExtensions
  SOURCES
    TinyFusionExtension.cpp
  EMBED_CAPI_LINK_LIBS
    TinyFusionCAPI
  PRIVATE_LINK_LIBS
    LLVMSupport
)

add_mlir_python_common_capi_library(TinyFusionCAPI
  INSTALL_COMPONENT MLIRTinyFusionPythonModules
  INSTALL_DESTINATION python_packages/mlir_tinyfusion/mlir/_mlir_libs
  OUTPUT_DIRECTORY "${MLIR_HLO_BINARY_DIR}/python_packages/mlir_tinyfusion/mlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../.."
  DECLARED_SOURCES
    MLIRPythonSources
    MLIRPythonExtension.RegisterEverything
    MLIRTinyFusionPythonSources
    MLIRTinyFusionPythonExtensions
)

add_mlir_python_modules(MLIRTinyFusionPythonModules
  ROOT_PREFIX "${MLIR_HLO_BINARY_DIR}/python_packages/mlir_tinyfusion/mlir"
  INSTALL_PREFIX "python_packages/mlir_tinyfusion/mlir"
  DECLARED_SOURCES
    MLIRPythonSources
    MLIRPythonExtension.RegisterEverything
    MLIRTinyFusionPythonSources
    MLIRTinyFusionPythonExtensions
  COMMON_CAPI_LINK_LIBS
    TinyFusionCAPI
  )