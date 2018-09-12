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

#
# C/C++ flags
#

CFLAGS    = -Wall #-std=c99 
CPPFLAGS  = -Wall

#
# CUDA flags
#

NVARCH	   = 35
NVFLAGS    = -arch=sm_$(NVARCH) -lcublas -lcurand

#
# Files to compile: 
#

MAIN   = main.cu 
CODC   = files.c
CODCPP = #files.cpp
CODCU  = #ngen-kutral.cu setup.cu

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

#
# The MAGIC
#

all:  $(BIN)$(EXE)

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

#
# Makefile for cleaning
# 

clean:
	$(RM) -rf $(DEST)*.o
	$(RM) -rf $(DEST)$(FC)*.o
	$(RM) -rf $(DEST)$(FCPP)*.o
	$(RM) -rf $(DEST)$(FCU)*.o

fresh:
	$(RM) -rf outputs/*

distclean: clean
	$(RM) -rf $(BIN)*