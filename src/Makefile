
include		../Makefile.inc

main: clean beamFormer tuning
 
beamFormer: beamFormer.cpp
	rm -f $(PROJ_BASE)/bin/beamFormer
	$(LINKER) -o $(PROJ_BASE)/bin/beamFormer $(INCLUDES) $(LIBS) $(HOME)/src/utils/src/utils.cpp beamFormer.cpp $(CFLAGS) $(LDFLAGS) 

tuning: tuning.cpp
	rm -f $(PROJ_BASE)/bin/tuning
	$(LINKER) -o $(PROJ_BASE)/bin/tuning $(INCLUDES) $(LIBS) $(HOME)/src/utils/src/utils.cpp tuning.cpp $(CFLAGS) $(LDFLAGS) 

clean:
	rm -f $(PROJ_BASE)/bin/beamFormer
	rm -f $(PROJ_BASE)/bin/tuning

