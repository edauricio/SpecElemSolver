CC=g++
IDIR = ../inc
CFLAGS = -std=c++11

ODIR = ../obj
SRCDIR = ../src

_DEPS = Poly.h Galerkin.h Fem.h Tensor.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ = poly.o galerkin.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))


MAIN = test
EXEC = test

main: check $(OBJ) $(MAIN)
	$(CC) $(ODIR)/$(MAIN).o $(OBJ) -o $(EXEC) -I$(IDIR) $(CFLAGS)

check:
	@mkdir -p -- $(ODIR)

$(MAIN): % : %.cpp $(DEPS)
	$(CC) -c $@.cpp -o $(ODIR)/$@.o -I$(IDIR) $(CFLAGS)

$(ODIR)/%.o: $(SRCDIR)/%.cpp $(DEPS)
	$(CC) -c $(@:$(ODIR)/%.o=$(SRCDIR)/%.cpp) -o $@ -I$(IDIR) $(CFLAGS)

.PHONY: clean

clean:
	-rm $(EXEC)
	-rm $(ODIR)/*.o
	-rm -rf $(ODIR)
