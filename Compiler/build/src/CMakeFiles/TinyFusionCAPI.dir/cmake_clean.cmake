file(REMOVE_RECURSE
  "libTinyFusionCAPI.a"
  "libTinyFusionCAPI.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/TinyFusionCAPI.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
