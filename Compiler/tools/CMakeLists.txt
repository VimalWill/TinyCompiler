get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRTosaDialect
        MLIROptLib
        MLIRTinyFusion
        )


add_llvm_executable(TinyCompiler-Opt TinyCompiler-Opt.cpp)
llvm_update_compile_flags(TinyCompiler-Opt)
target_link_libraries(TinyCompiler-Opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(TinyCompiler-Opt)