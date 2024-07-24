add_mlir_dialect_library(MLIRTinyFusion
        TinyFusionDialect.cpp
        LowerToTinyFusion.cpp 

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/includes

        DEPENDS
        MLIRTinyFusionIncGen

        LINK_LIBS PUBLIC

        MLIRAnalysis
        MLIRCallInterfaces
        MLIRCastInterfaces
        MLIRExecutionEngine
        MLIRIR
        MLIRTosaDialect
        MLIRLLVMCommonConversion
        MLIRLLVMDialect
        MLIRLLVMToLLVMIRTranslation
        MLIRMemRefDialect
        MLIRParser
        MLIRPass
        MLIRSideEffectInterfaces
        MLIRSupport
        MLIRTargetLLVMIRExport
        MLIRTransforms
        )