CC=g++
IDIR = ../inc
CGLAFS = -I$(IDIR)

ODIR = ../obj
SRCDIR = ../src

_DEPS = Poly.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ = poly.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

SRC = $(patsubst %,$(SRCDIR)/%,$(_SRC))

MAIN = test
EXEC = test

main: check $(OBJ) $(MAIN)
	$(CC) $(ODIR)/$(MAIN).o $(OBJ) -o $(EXEC) -I$(IDIR)

check:
	@mkdir -p -- $(ODIR)

$(MAIN): $(DEPS)
	$(CC) -c $@.cpp -o $(ODIR)/$@.o -I$(IDIR)

$(ODIR)/%.o: $(DEPS)
	$(CC) -c $(@:$(ODIR)/%.o=$(SRCDIR)/%.cpp) -o $@ -I$(IDIR)

.PHONY: clean

clean:
	-rm $(EXEC)
	-rm $(ODIR)/*.o
	-rm -rf $(ODIR)