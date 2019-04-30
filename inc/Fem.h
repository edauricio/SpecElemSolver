class Element {
public:
	// Constructors
	Element(size_t ver, size_t ed, size_t fa) : v(ver), e(ed), f(fa) {}

	// Member functions
	size_t vsize() { return v; }
	size_t esize() { return e; }
	size_t fsize() { return f; }
	virtual size_t nprinc() = 0;


protected:
	size_t v, e, f;
};

class Line : public Element {
public:
	// Types
	typedef Matrix PrincFuncStruct;

	// Constructor
	Line() : Element(2, 1, 0) {};

	// Member functions
	virtual size_t nprinc() { return 1; }
};

class Tri : public Element {
public:
	Tri() : Element(3, 3, 1) {};
	virtual size_t nprinc() { return 2; }
};

class Quad : public Element {
public:
	Quad() : Element(4, 4, 1) {};
	virtual size_t nprinc() { return 1; }
};

class Hexa : public Element {
public:
	Hexa() : Element (8, 12, 6) {};
	~Hexa() { delete[] faces; }

protected:
	Quad *faces = new Quad[6];
};