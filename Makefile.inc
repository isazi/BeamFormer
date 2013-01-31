
PROJ_BASE	:= $(HOME)/src/BeamFormer

INCLUDES	:= -I"$(PROJ_BASE)/include" -I"${OPENCL_INCLUDE}" -I"$(HOME)/src/OpenCL/include" -I"$(HOME)/src/utils/include"
LIBS		:= -L"${OPENCL_LIB}"

ifneq ($(debug), 1)
	CFLAGS		:= -std=c++0x -O3 -g0 -w -msse3 -fopenmp
else
	CFLAGS		:= -std=c++0x -O0 -g3 -Wall -msse3 -fopenmp
endif

LDFLAGS         := -lm -lOpenCL

CC		:= g++
LINKER		:= g++

