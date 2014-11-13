
PROJ_BASE	:= $(HOME)/src/BeamFormer

# https://github.com/isazi/utils
ISA_UTILS_INCLUDE	:= $(HOME)/src/utils/include
# https://github.com/isazi/OpenCL
ISA_OPENCL_INCLUDE	:= $(HOME)/src/OpenCL/include
# https://github.com/isazi/AstroData
ISA_ASTRO_INCLUDE	:= $(HOME)/src/AstroData/include

INCLUDES := -I"$(PROJ_BASE)/include" -I"$(OPENCL_INCLUDE)" -I"$(ISA_UTILS_INCLUDE)" -I"$(ISA_OPENCL_INCLUDE)" -I"$(ISA_ASTRO_INCLUDE)"
LIBS := -L"$(OPENCL_LIB)"

ifneq ($(debug), 1)
	CFLAGS := -std=c++0x -O3 -g0 -Wall -fopenmp
else
	CFLAGS := -std=c++0x -O0 -g3 -Wall -fopenmp
endif

LDFLAGS := -lm -lOpenCL

CC := icc

