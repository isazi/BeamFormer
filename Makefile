
SUBDIRS := src

all:
	@for dir in $(SUBDIRS); do\
		echo "Making all in $$dir.";\
		(cd $$dir; make all); done

clean:
	@for dir in $(SUBDIRS); do\
		echo "Cleaning all in $$dir.";\
		(cd $$dir; make clean); done

