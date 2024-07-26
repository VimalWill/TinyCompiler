add_mlir_dialect_library(MLIRTinyFusion
        TinyFusionDialect.cpp
        LowerToTinyFusion.cpp 

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/includes

        PARTIAL_SOURCES_INTENDED

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

add_mlir_public_c_api_library(TinyFusionCAPI
        TinyFusionCApi.cpp
        
        # to silence LLVMProcessSources.cmake:116
        
        LINK_LIBS PUBLIC
        MLIRTinyFusion
        )