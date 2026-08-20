CXX ?= g++
CXXFLAGS ?= -O3 -march=native -fopenmp -funroll-loops -fomit-frame-pointer -flto -std=c++17 -Wall -Wextra
SRC_EXE = src/main.cpp
SRC_LIB = src/parasat_api.cpp
BIN_DIR = bin

TARGET_EXE = $(BIN_DIR)/parasat_gpu
TARGET_LIB = $(BIN_DIR)/libparasat.so

ifeq ($(OS),Windows_NT)
    TARGET_EXE := $(BIN_DIR)/parasat_gpu.exe
    TARGET_LIB := $(BIN_DIR)/parasat.dll
    MKDIR = if not exist $(BIN_DIR) mkdir $(BIN_DIR)
    RM = del /Q /F
else
    MKDIR = mkdir -p $(BIN_DIR)
    RM = rm -rf
endif

all: $(TARGET_EXE) $(TARGET_LIB)

$(TARGET_EXE): $(SRC_EXE) src/*.hpp
	@$(MKDIR)
	$(CXX) $(CXXFLAGS) -Isrc $(SRC_EXE) -o $(TARGET_EXE)
	@echo [ParaSAT-GPU] Executable built: $(TARGET_EXE)

$(TARGET_LIB): $(SRC_LIB) src/*.hpp include/parasat.h
	@$(MKDIR)
	$(CXX) $(CXXFLAGS) -shared -fPIC -Isrc -Iinclude $(SRC_LIB) -o $(TARGET_LIB)
	@echo [ParaSAT-GPU] Shared Library built: $(TARGET_LIB)

clean:
	$(RM) $(TARGET_EXE) $(TARGET_LIB)

.PHONY: all clean
