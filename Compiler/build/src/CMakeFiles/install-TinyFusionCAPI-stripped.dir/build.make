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

# Utility rule file for install-TinyFusionCAPI-stripped.

# Include any custom commands dependencies for this target.
include src/CMakeFiles/install-TinyFusionCAPI-stripped.dir/compiler_depend.make

# Include the progress variables for this target.
include src/CMakeFiles/install-TinyFusionCAPI-stripped.dir/progress.make

src/CMakeFiles/install-TinyFusionCAPI-stripped:
	cd /home/vimal/TinyCompiler/Compiler/build/src && /home/vimal/.local/lib/python3.10/site-packages/cmake/data/bin/cmake -DCMAKE_INSTALL_COMPONENT="TinyFusionCAPI" -DCMAKE_INSTALL_DO_STRIP=1 -P /home/vimal/TinyCompiler/Compiler/build/cmake_install.cmake

install-TinyFusionCAPI-stripped: src/CMakeFiles/install-TinyFusionCAPI-stripped
install-TinyFusionCAPI-stripped: src/CMakeFiles/install-TinyFusionCAPI-stripped.dir/build.make
.PHONY : install-TinyFusionCAPI-stripped

# Rule to build all files generated by this target.
src/CMakeFiles/install-TinyFusionCAPI-stripped.dir/build: install-TinyFusionCAPI-stripped
.PHONY : src/CMakeFiles/install-TinyFusionCAPI-stripped.dir/build

src/CMakeFiles/install-TinyFusionCAPI-stripped.dir/clean:
	cd /home/vimal/TinyCompiler/Compiler/build/src && $(CMAKE_COMMAND) -P CMakeFiles/install-TinyFusionCAPI-stripped.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/install-TinyFusionCAPI-stripped.dir/clean

src/CMakeFiles/install-TinyFusionCAPI-stripped.dir/depend:
	cd /home/vimal/TinyCompiler/Compiler/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/vimal/TinyCompiler/Compiler /home/vimal/TinyCompiler/Compiler/src /home/vimal/TinyCompiler/Compiler/build /home/vimal/TinyCompiler/Compiler/build/src /home/vimal/TinyCompiler/Compiler/build/src/CMakeFiles/install-TinyFusionCAPI-stripped.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : src/CMakeFiles/install-TinyFusionCAPI-stripped.dir/depend
