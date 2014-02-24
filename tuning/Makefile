
include		../Makefile.inc

main: clean BeamFormer
 
tuning: BeamFormer.cpp
	$(LINKER) -o $(PROJ_BASE)/bin/BeamFormerTuning BeamFormer.cpp $(UTILS_SRC) $(INCLUDES) $(LIBS) $(CFLAGS) $(LDFLAGS)

clean:
	rm -f $(PROJ_BASE)/bin/BeamFormerTuning
