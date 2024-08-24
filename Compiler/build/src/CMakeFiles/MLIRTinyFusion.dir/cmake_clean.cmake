file(REMOVE_RECURSE
  "libMLIRTinyFusion.a"
  "libMLIRTinyFusion.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/MLIRTinyFusion.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
