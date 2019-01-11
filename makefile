#
# Folder structure
#

BIN     := ./bin/
SRC     := ./src/
DEST    := ./obj/

EXE    = ngen-kutral

#
# Executables
#

CC     = g++#gcc
CP     = g++
NVCC   = nvcc
RM     = rm
MKDIR  = mkdir -p

#
# C/C++ flags
#

CFLAGS    = -Wall #-std=c99 
CPPFLAGS  = -Wall

#
# CUDA flags
#

NVARCH	   = 35
NVFLAGS    = -G -g -arch=sm_$(NVARCH) -lcurand #-lcublas 

#
# Files to compile: 
#

MAIN   = main.cu 
CODC   = files.c diffmat.c utils.c
CODCPP = #files.cpp
CODCU  = diffmat.cu wildfire.cu utils.cu #linalg.cu

#
# Formating the folder structure for compiling/linking/cleaning.
#

FC     = c/
FCPP   = cpp/
FCU    = cu/

#
# Preparing variables for automated prerequisites
#

OBJC   = $(patsubst %.c,$(DEST)$(FC)%.o,$(CODC))
OBJCPP = $(patsubst %.cpp,$(DEST)$(FCPP)%.o,$(CODCPP))
OBJCU  = $(patsubst %.cu,$(DEST)$(FCU)%.o,$(CODCU))

SRCMAIN = $(patsubst %,$(SRC)%,$(MAIN))
OBJMAIN = $(patsubst $(SRC)%.cu,$(DEST)%.o,$(SRCMAIN))

# .PHONY: directories

#
# The MAGIC
#

all: directories $(BIN)$(EXE) clean

$(BIN)$(EXE): $(OBJC) $(OBJCPP) $(OBJCU) $(OBJMAIN)
	$(NVCC) $(NVFLAGS) $^ -o $@

$(OBJMAIN): $(SRCMAIN)
	$(NVCC) $(NVFLAGS) -dc $? -o $@

$(OBJCPP): $(DEST)%.o : $(SRC)%.cpp
	$(CP) $(CPPFLAGS) -c $? -o $@

$(OBJC): $(DEST)%.o : $(SRC)%.c
	$(CC) $(CFLAGS) -c $? -o $@

$(OBJCU): $(DEST)%.o : $(SRC)%.cu
	$(NVCC) $(NVFLAGS) -dc $? -o $@


directories:
	$(MKDIR) $(BIN)
	$(MKDIR) $(DEST)$(FC) 
	$(MKDIR) $(DEST)$(FCPP)
	$(MKDIR) $(DEST)$(FCU)  
#
# Makefile for cleaning
# 

clean:
	$(RM) -rf $(DEST)*.o
	$(RM) -rf $(DEST)$(FC)*.o
	$(RM) -rf $(DEST)$(FCPP)*.o
	$(RM) -rf $(DEST)$(FCU)*.o
	$(RM) -rf $(DEST)

fresh:
	$(RM) -rf outputs/*

distclean: clean
	$(RM) -rf $(BIN)*