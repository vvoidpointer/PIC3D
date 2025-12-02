# Makefile for PIC2D Simulation

# Compiler
CXX = mpicxx

# Compiler flags
CXXFLAGS = -std=c++11 -I./include -Wall -g

# HDF5 Library flags
HDF5_LDFLAGS = -L/opt/ohpc/pub/libs/gnu12/openmpi4/hdf5/1.10.8/lib  -lhdf5

# HDF5 头文件路径（编译时需要）
HDF5_CFLAGS = -I/opt/ohpc/pub/libs/gnu12/openmpi4/hdf5/1.10.8/include

# 将 HDF5 头文件路径加入编译选项
CXXFLAGS += $(HDF5_CFLAGS)

# Directories
BUILD_DIR = build

# Source files
SRCS = $(wildcard src/*.cpp) main.cpp

# Object files - place them in the build directory
OBJS = $(patsubst src/%.cpp,$(BUILD_DIR)/%.o,$(wildcard src/*.cpp))
OBJS += $(patsubst %.cpp,$(BUILD_DIR)/%.o,main.cpp)

# Executable name
TARGET = pic3d

# Default target
all: $(TARGET)

# Linking the executable
$(TARGET): $(OBJS)
	@echo "Linking..."
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS) $(HDF5_LDFLAGS)

# Rule for compiling source files from src/ directory
$(BUILD_DIR)/%.o: src/%.cpp
	@mkdir -p $(BUILD_DIR)
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule for compiling main.cpp
$(BUILD_DIR)/main.o: main.cpp
	@mkdir -p $(BUILD_DIR)
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up build files
clean:
	@echo "Cleaning up..."
	rm -rf $(BUILD_DIR) $(TARGET)

.PHONY: all clean

