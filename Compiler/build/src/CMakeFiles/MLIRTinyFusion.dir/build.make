# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.27

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/vimal/.local/lib/python3.10/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /home/vimal/.local/lib/python3.10/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/vimal/TinyCompiler/Compiler

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/vimal/TinyCompiler/Compiler/build

# Include any dependencies generated for this target.
include src/CMakeFiles/MLIRTinyFusion.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/CMakeFiles/MLIRTinyFusion.dir/compiler_depend.make

# Include the progress variables for this target.
include src/CMakeFiles/MLIRTinyFusion.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/MLIRTinyFusion.dir/flags.make

# Object files for target MLIRTinyFusion
MLIRTinyFusion_OBJECTS =

# External object files for target MLIRTinyFusion
MLIRTinyFusion_EXTERNAL_OBJECTS = \
"/home/vimal/TinyCompiler/Compiler/build/src/CMakeFiles/obj.MLIRTinyFusion.dir/TinyFusionDialect.cpp.o" \
"/home/vimal/TinyCompiler/Compiler/build/src/CMakeFiles/obj.MLIRTinyFusion.dir/LowerToTinyFusion.cpp.o"

src/libMLIRTinyFusion.a: src/CMakeFiles/obj.MLIRTinyFusion.dir/TinyFusionDialect.cpp.o
src/libMLIRTinyFusion.a: src/CMakeFiles/obj.MLIRTinyFusion.dir/LowerToTinyFusion.cpp.o
src/libMLIRTinyFusion.a: src/CMakeFiles/MLIRTinyFusion.dir/build.make
src/libMLIRTinyFusion.a: src/CMakeFiles/MLIRTinyFusion.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/vimal/TinyCompiler/Compiler/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Linking CXX static library libMLIRTinyFusion.a"
	cd /home/vimal/TinyCompiler/Compiler/build/src && $(CMAKE_COMMAND) -P CMakeFiles/MLIRTinyFusion.dir/cmake_clean_target.cmake
	cd /home/vimal/TinyCompiler/Compiler/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/MLIRTinyFusion.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/MLIRTinyFusion.dir/build: src/libMLIRTinyFusion.a
.PHONY : src/CMakeFiles/MLIRTinyFusion.dir/build

src/CMakeFiles/MLIRTinyFusion.dir/clean:
	cd /home/vimal/TinyCompiler/Compiler/build/src && $(CMAKE_COMMAND) -P CMakeFiles/MLIRTinyFusion.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/MLIRTinyFusion.dir/clean

src/CMakeFiles/MLIRTinyFusion.dir/depend:
	cd /home/vimal/TinyCompiler/Compiler/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/vimal/TinyCompiler/Compiler /home/vimal/TinyCompiler/Compiler/src /home/vimal/TinyCompiler/Compiler/build /home/vimal/TinyCompiler/Compiler/build/src /home/vimal/TinyCompiler/Compiler/build/src/CMakeFiles/MLIRTinyFusion.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : src/CMakeFiles/MLIRTinyFusion.dir/depend

