# Makefile

# Compiler
NVCC = nvcc

# Executable name
TARGET = heat_cuda

# Source file
SRC = heat_cuda.cu

# Default target
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) -o $(TARGET) $(SRC)

# Clean rule to remove compiled files
clean:
	rm -f $(TARGET)