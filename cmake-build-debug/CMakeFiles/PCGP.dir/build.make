# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = C:\Users\bruno\AppData\Local\JetBrains\Toolbox\apps\CLion\ch-0\201.7223.86\bin\cmake\win\bin\cmake.exe

# The command to remove a file.
RM = C:\Users\bruno\AppData\Local\JetBrains\Toolbox\apps\CLion\ch-0\201.7223.86\bin\cmake\win\bin\cmake.exe -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = C:\Users\bruno\CLionProjects\P-CGPDE

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = C:\Users\bruno\CLionProjects\P-CGPDE\cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/PCGP.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/PCGP.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/PCGP.dir/flags.make

CMakeFiles/PCGP.dir/main.cpp.obj: CMakeFiles/PCGP.dir/flags.make
CMakeFiles/PCGP.dir/main.cpp.obj: CMakeFiles/PCGP.dir/includes_CXX.rsp
CMakeFiles/PCGP.dir/main.cpp.obj: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\bruno\CLionProjects\P-CGPDE\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/PCGP.dir/main.cpp.obj"
	C:\mingw64\bin\g++.exe  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\PCGP.dir\main.cpp.obj -c C:\Users\bruno\CLionProjects\P-CGPDE\main.cpp

CMakeFiles/PCGP.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/PCGP.dir/main.cpp.i"
	C:\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\bruno\CLionProjects\P-CGPDE\main.cpp > CMakeFiles\PCGP.dir\main.cpp.i

CMakeFiles/PCGP.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/PCGP.dir/main.cpp.s"
	C:\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\bruno\CLionProjects\P-CGPDE\main.cpp -o CMakeFiles\PCGP.dir\main.cpp.s

CMakeFiles/PCGP.dir/circuit.cpp.obj: CMakeFiles/PCGP.dir/flags.make
CMakeFiles/PCGP.dir/circuit.cpp.obj: CMakeFiles/PCGP.dir/includes_CXX.rsp
CMakeFiles/PCGP.dir/circuit.cpp.obj: ../circuit.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\bruno\CLionProjects\P-CGPDE\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/PCGP.dir/circuit.cpp.obj"
	C:\mingw64\bin\g++.exe  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\PCGP.dir\circuit.cpp.obj -c C:\Users\bruno\CLionProjects\P-CGPDE\circuit.cpp

CMakeFiles/PCGP.dir/circuit.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/PCGP.dir/circuit.cpp.i"
	C:\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\bruno\CLionProjects\P-CGPDE\circuit.cpp > CMakeFiles\PCGP.dir\circuit.cpp.i

CMakeFiles/PCGP.dir/circuit.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/PCGP.dir/circuit.cpp.s"
	C:\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\bruno\CLionProjects\P-CGPDE\circuit.cpp -o CMakeFiles\PCGP.dir\circuit.cpp.s

CMakeFiles/PCGP.dir/utils.cpp.obj: CMakeFiles/PCGP.dir/flags.make
CMakeFiles/PCGP.dir/utils.cpp.obj: CMakeFiles/PCGP.dir/includes_CXX.rsp
CMakeFiles/PCGP.dir/utils.cpp.obj: ../utils.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\bruno\CLionProjects\P-CGPDE\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/PCGP.dir/utils.cpp.obj"
	C:\mingw64\bin\g++.exe  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\PCGP.dir\utils.cpp.obj -c C:\Users\bruno\CLionProjects\P-CGPDE\utils.cpp

CMakeFiles/PCGP.dir/utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/PCGP.dir/utils.cpp.i"
	C:\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\bruno\CLionProjects\P-CGPDE\utils.cpp > CMakeFiles\PCGP.dir\utils.cpp.i

CMakeFiles/PCGP.dir/utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/PCGP.dir/utils.cpp.s"
	C:\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\bruno\CLionProjects\P-CGPDE\utils.cpp -o CMakeFiles\PCGP.dir\utils.cpp.s

CMakeFiles/PCGP.dir/bitwise.cpp.obj: CMakeFiles/PCGP.dir/flags.make
CMakeFiles/PCGP.dir/bitwise.cpp.obj: CMakeFiles/PCGP.dir/includes_CXX.rsp
CMakeFiles/PCGP.dir/bitwise.cpp.obj: ../bitwise.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\bruno\CLionProjects\P-CGPDE\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/PCGP.dir/bitwise.cpp.obj"
	C:\mingw64\bin\g++.exe  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\PCGP.dir\bitwise.cpp.obj -c C:\Users\bruno\CLionProjects\P-CGPDE\bitwise.cpp

CMakeFiles/PCGP.dir/bitwise.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/PCGP.dir/bitwise.cpp.i"
	C:\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\bruno\CLionProjects\P-CGPDE\bitwise.cpp > CMakeFiles\PCGP.dir\bitwise.cpp.i

CMakeFiles/PCGP.dir/bitwise.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/PCGP.dir/bitwise.cpp.s"
	C:\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\bruno\CLionProjects\P-CGPDE\bitwise.cpp -o CMakeFiles\PCGP.dir\bitwise.cpp.s

CMakeFiles/PCGP.dir/stack.cpp.obj: CMakeFiles/PCGP.dir/flags.make
CMakeFiles/PCGP.dir/stack.cpp.obj: CMakeFiles/PCGP.dir/includes_CXX.rsp
CMakeFiles/PCGP.dir/stack.cpp.obj: ../stack.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\bruno\CLionProjects\P-CGPDE\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/PCGP.dir/stack.cpp.obj"
	C:\mingw64\bin\g++.exe  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\PCGP.dir\stack.cpp.obj -c C:\Users\bruno\CLionProjects\P-CGPDE\stack.cpp

CMakeFiles/PCGP.dir/stack.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/PCGP.dir/stack.cpp.i"
	C:\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\bruno\CLionProjects\P-CGPDE\stack.cpp > CMakeFiles\PCGP.dir\stack.cpp.i

CMakeFiles/PCGP.dir/stack.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/PCGP.dir/stack.cpp.s"
	C:\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\bruno\CLionProjects\P-CGPDE\stack.cpp -o CMakeFiles\PCGP.dir\stack.cpp.s

CMakeFiles/PCGP.dir/GPTime.cpp.obj: CMakeFiles/PCGP.dir/flags.make
CMakeFiles/PCGP.dir/GPTime.cpp.obj: CMakeFiles/PCGP.dir/includes_CXX.rsp
CMakeFiles/PCGP.dir/GPTime.cpp.obj: ../GPTime.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\bruno\CLionProjects\P-CGPDE\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/PCGP.dir/GPTime.cpp.obj"
	C:\mingw64\bin\g++.exe  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\PCGP.dir\GPTime.cpp.obj -c C:\Users\bruno\CLionProjects\P-CGPDE\GPTime.cpp

CMakeFiles/PCGP.dir/GPTime.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/PCGP.dir/GPTime.cpp.i"
	C:\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\bruno\CLionProjects\P-CGPDE\GPTime.cpp > CMakeFiles\PCGP.dir\GPTime.cpp.i

CMakeFiles/PCGP.dir/GPTime.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/PCGP.dir/GPTime.cpp.s"
	C:\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\bruno\CLionProjects\P-CGPDE\GPTime.cpp -o CMakeFiles\PCGP.dir\GPTime.cpp.s

# Object files for target PCGP
PCGP_OBJECTS = \
"CMakeFiles/PCGP.dir/main.cpp.obj" \
"CMakeFiles/PCGP.dir/circuit.cpp.obj" \
"CMakeFiles/PCGP.dir/utils.cpp.obj" \
"CMakeFiles/PCGP.dir/bitwise.cpp.obj" \
"CMakeFiles/PCGP.dir/stack.cpp.obj" \
"CMakeFiles/PCGP.dir/GPTime.cpp.obj"

# External object files for target PCGP
PCGP_EXTERNAL_OBJECTS =

PCGP.exe: CMakeFiles/PCGP.dir/main.cpp.obj
PCGP.exe: CMakeFiles/PCGP.dir/circuit.cpp.obj
PCGP.exe: CMakeFiles/PCGP.dir/utils.cpp.obj
PCGP.exe: CMakeFiles/PCGP.dir/bitwise.cpp.obj
PCGP.exe: CMakeFiles/PCGP.dir/stack.cpp.obj
PCGP.exe: CMakeFiles/PCGP.dir/GPTime.cpp.obj
PCGP.exe: CMakeFiles/PCGP.dir/build.make
PCGP.exe: CMakeFiles/PCGP.dir/linklibs.rsp
PCGP.exe: CMakeFiles/PCGP.dir/objects1.rsp
PCGP.exe: CMakeFiles/PCGP.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=C:\Users\bruno\CLionProjects\P-CGPDE\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Linking CXX executable PCGP.exe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\PCGP.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/PCGP.dir/build: PCGP.exe

.PHONY : CMakeFiles/PCGP.dir/build

CMakeFiles/PCGP.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\PCGP.dir\cmake_clean.cmake
.PHONY : CMakeFiles/PCGP.dir/clean

CMakeFiles/PCGP.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" C:\Users\bruno\CLionProjects\P-CGPDE C:\Users\bruno\CLionProjects\P-CGPDE C:\Users\bruno\CLionProjects\P-CGPDE\cmake-build-debug C:\Users\bruno\CLionProjects\P-CGPDE\cmake-build-debug C:\Users\bruno\CLionProjects\P-CGPDE\cmake-build-debug\CMakeFiles\PCGP.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/PCGP.dir/depend

