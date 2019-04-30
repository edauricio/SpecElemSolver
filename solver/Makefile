CC=g++
IDIR = ../inc
CGLAFS = -I$(IDIR)

ODIR = ../obj
SRCDIR = ../src

_DEPS = Poly.h Galerkin.h Fem.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ = poly.o galerkin.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))


MAIN = test
EXEC = test

main: check $(OBJ) $(MAIN)
	$(CC) $(ODIR)/$(MAIN).o $(OBJ) -o $(EXEC) -I$(IDIR)

check:
	@mkdir -p -- $(ODIR)

$(MAIN): % : %.cpp $(DEPS)
	$(CC) -c $@.cpp -o $(ODIR)/$@.o -I$(IDIR)

$(ODIR)/%.o: $(SRCDIR)/%.cpp $(DEPS)
	$(CC) -c $(@:$(ODIR)/%.o=$(SRCDIR)/%.cpp) -o $@ -I$(IDIR)

.PHONY: clean

clean:
	-rm $(EXEC)
	-rm $(ODIR)/*.o
	-rm -rf $(ODIR)
