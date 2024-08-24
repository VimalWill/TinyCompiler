/**
 * the file includes the main function for the tiny-compiler
 * and the pass-manager pipeline which lower tosa to the llvm
 */

#include "Dialect/Passes.h"
#include "Dialect/TinyFusionDialect.h"

#include "llvm/Support/raw_ostream.h"

int main(int argc, char *argv[]) {

  if (argc == 0) {
    llvm::errs() << "expected atleast one input\n";
    return 0;
  }

  int count args = 0;
  while (args < argc) {
    /*check for .mlir file*/
  }

  return 0;
}