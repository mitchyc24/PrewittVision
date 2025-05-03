CC=gcc
NVCC=nvcc

# Define directories
SRC_DIR = src
INCLUDE_DIR = include
BUILD_DIR = build
LOG_DIR = logs
DATA_DIR = data
OUTPUT_DIR = output

# Compiler flags
CFLAGS=-I$(INCLUDE_DIR) -fopenmp
NVFLAGS=-I$(INCLUDE_DIR) -L/usr/local/cuda/lib64 -lcudart -ccbin g++ -Xcompiler -fopenmp

# Source files
C_SRCS = $(wildcard $(SRC_DIR)/*.c)
CU_SRCS = $(wildcard $(SRC_DIR)/*.cu)

# Object files (place them in the build directory)
OBJ = $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.o, $(filter-out $(SRC_DIR)/lodepng.c, $(C_SRCS))) $(BUILD_DIR)/lodepng.o $(patsubst $(SRC_DIR)/%.cu, $(BUILD_DIR)/%.o, $(CU_SRCS))

# Header files (dependencies)
DEPS = $(wildcard $(INCLUDE_DIR)/*.h)

# Executable name
TARGET = PrewittVision

# Default target
all: $(TARGET)

# Rule to build C object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c $(DEPS)
	@mkdir -p $(BUILD_DIR) # Ensure build directory exists
	$(CC) -c -o $@ $< $(CFLAGS)

# Special rule for lodepng.c
$(BUILD_DIR)/lodepng.o: $(SRC_DIR)/lodepng.c $(INCLUDE_DIR)/lodepng.h
	@mkdir -p $(BUILD_DIR) # Ensure build directory exists
	$(CC) -c -o $@ $< $(CFLAGS)

# Rule to build CUDA object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu $(DEPS)
	@mkdir -p $(BUILD_DIR) # Ensure build directory exists
	$(NVCC) -c -o $@ $< $(NVFLAGS)

# Rule to link the executable
$(TARGET): $(OBJ)
	$(NVCC) -o $@ $^ $(NVFLAGS)

.PHONY: clean run

# Rule to clean build artifacts
clean:
	rm -rf $(BUILD_DIR) $(TARGET) $(LOG_DIR)/* $(OUTPUT_DIR)/*

# Optional: Rule to run the executable (assuming it takes input dir and output dir)
run: $(TARGET)
	@mkdir -p $(OUTPUT_DIR) # Ensure output directory exists
	./$(TARGET) $(DATA_DIR) $(OUTPUT_DIR)
