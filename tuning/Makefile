
include		../Makefile.inc

all: clean BeamFormer
 
BeamFormer: BeamFormer.cpp
	$(CC) -o $(PROJ_BASE)/bin/BeamFormerTuning BeamFormer.cpp $(INCLUDES) $(LIBS) $(CFLAGS) $(LDFLAGS)

clean:
	rm -f $(PROJ_BASE)/bin/BeamFormerTuning
