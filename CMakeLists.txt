cmake_minimum_required(VERSION 3.10)
project(PIC3D VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Find HDF5
find_package(HDF5 REQUIRED COMPONENTS CXX)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${HDF5_INCLUDE_DIRS})

# Source files
set(SOURCES
    src/main.cpp
    src/particle.cpp
    src/field.cpp
    src/grid.cpp
    src/io.cpp
)

# Header files
set(HEADERS
    include/particle.h
    include/field.h
    include/grid.h
    include/io.h
    include/constants.h
)

# Create executable
add_executable(pic3d ${SOURCES} ${HEADERS})

# Link libraries
target_link_libraries(pic3d 
    OpenMP::OpenMP_CXX
    ${HDF5_LIBRARIES}
    ${HDF5_CXX_LIBRARIES}
)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(pic3d PRIVATE -Wall -Wextra -O3)
endif()

# Installation
install(TARGETS pic3d DESTINATION bin)
