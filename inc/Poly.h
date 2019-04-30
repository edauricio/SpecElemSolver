#ifndef POLY_H
#define POLY_H

#include <iterator>
#include <string>
#include <cstddef>
#include <map>
#include <initializer_list>

template <typename T>
class Iterator : std::iterator<std::random_access_iterator_tag, T> {
public:
  // Single Constructor
  Iterator(T* obj) : it(obj) {};

  // Operator overloading
  T& operator*() { return *it; };
  Iterator operator+(size_t s) { return this->it + s; }
  Iterator& operator++() { it += 1; return *this; };
  Iterator operator++(int) { it += 1; return this->it - 1; };
  std::ptrdiff_t operator-(const Iterator& it2) { return it - it2.it; }
  Iterator operator-(int s) { return it-s; }
  Iterator& operator--() { it -= 1; return *this; };
  Iterator operator--(int) { it -= 1; return this->it + 1; };
  bool operator==(const Iterator& it2) { return (this->it == it2.it); };
  bool operator!=(const Iterator& it2) { return (this->it != it2.it); };

private:
  T* it;
};

class Tensor {
public:
  // Types
  typedef Iterator<double> iterator;

  // Constructors
  Tensor() {};
  explicit Tensor(size_t qt) : quantity(qt), usage(new size_t(1)), points(new double[quantity]), hasNew(true) {};
  Tensor(const Tensor&);
  Tensor(Tensor&&);

  // Destructors
  virtual ~Tensor();

  // Member functions
  virtual size_t size() { return quantity; };
  virtual size_t rsize() = 0;
  virtual size_t csize() = 0;
  bool empty() { return (!quantity); };
  virtual void resize(size_t, size_t) = 0;
  iterator begin() { return Iterator<double>(points); }
  iterator end() { return Iterator<double>(points+quantity); }
  void free();

  // Overloaded operators
  Tensor& operator=(Tensor&); // Copy-assignmet operator
  Tensor& operator=(Tensor&&); // Move-assignment operator

  virtual double& operator()(size_t = 0, size_t = 0, size_t = 0) {};

protected:
  size_t quantity = 0;
  size_t *usage = new size_t(1);
  double *points;
  bool hasNew = false;

  bool check_range(size_t);
};

class Vector : public Tensor {
public:
  // Types
  //typedef Iterator<double> iterator;

  // Constructors
  //using Tensor::Tensor; // Inheritance from base class
  Vector() {};
  explicit Vector(size_t qt) : Tensor(qt), n(qt) {};

  // Member functions
  virtual void resize(size_t, size_t = 0);
  virtual size_t rsize() { return n; }
  virtual size_t csize() { return 1; }

  // Overloaded operators
  virtual double& operator()(size_t, size_t = 0, size_t = 0);

private:
  size_t n;
};

class Matrix : public Tensor {
public:
  // Constructors
  Matrix() {};
  explicit Matrix(size_t i) : Tensor(i*i), n(i), m(i) {};
  Matrix(size_t i, size_t j) : Tensor(i*j), n(i), m(j) {};

  // Member functions
  virtual void resize(size_t, size_t);
  virtual size_t size() { if (n == m) return n; else throw std::out_of_range("matrix is not square; ambiguous size() call"); }
  virtual size_t rsize() { return n; }
  virtual size_t csize() { return m; }
  void resize(size_t nsz) { resize(nsz, nsz); }

  // Overlodad operators
  virtual double& operator()(size_t, size_t, size_t = 0);

private:
  size_t n, m;

};


class BasePoly {
public:
  // Types
  typedef Vector::iterator iterator;

  // Constructors
  explicit BasePoly(size_t ord) : P(ord), zeros(ord) { };

  // Member functions
  size_t order() { return P; };
  virtual double calcPoly(double) = 0;
  virtual double calcDeriv(double) = 0;
  Vector& Zeros() { return zeros; };

  // Iterator
  Vector::iterator zbegin() { return zeros.begin(); }
  Vector::iterator zend() { return zeros.end(); }

protected:  
  int P;
  Vector zeros;
  bool hasAB;

};

class Jacobian : public BasePoly {
  friend class GLL;

public:
  // Constructors
  explicit Jacobian(size_t ord, double a = 1., double b = 1.) : BasePoly(ord), Alpha(a), Beta(b) { hasAB = true; JacPZ(P, Alpha, Beta, zeros); };

  // Member functions
  double alpha() { return Alpha; };
  double beta() { return Beta; };
  virtual Jacobian& setAlpha(double a) { if (a != Alpha) { Alpha = a; clear_n_zeros(); } return *this; };
  virtual Jacobian& setBeta(double b) { if (b != Beta) { Beta = b; clear_n_zeros(); } return *this; };
  Jacobian& setOrder(size_t ord) { if (ord != P) { P = ord; clear_n_zeros(true); } return *this; };
  Jacobian& reset(size_t ord, double a, double b) { Alpha = a; Beta = b; if (ord != P) { P = ord; clear_n_zeros(true); } else clear_n_zeros(); }
  virtual double calcPoly(double x) { return JacP(P, Alpha, Beta, x); };
  virtual double calcDeriv(double x) { return dJacP(P, Alpha, Beta, x); };
  virtual std::string type() { return std::string("Jacobian"); }
  virtual Vector& Zeros() { return zeros; }
  bool isJac() { return hasAB; };

protected:
  double JacP(const size_t&, const double&, const double&, const double&);
  double dJacP(const size_t&, const double&, const double&, const double&);
  Tensor& JacPZ(const size_t&, const double&, const double&, Tensor&);
  void JacPZ(const size_t&, const double&, const double&, iterator, iterator);
  double Alpha, Beta;

private:
  void clear_n_zeros(bool = false);
};

class Legendre : public Jacobian {
public:
  explicit Legendre(size_t ord) : Jacobian(ord, 0, 0) {};
  virtual Legendre& setAlpha(double a) { throw std::out_of_range("Legendre polynomial; alpha is fixed."); };
  virtual Legendre& setBeta(double b) { throw std::out_of_range("Legendre polynomial; beta is fixed."); };
  virtual std::string type() { return std::string("Legendre"); }
};

class Chebyshev : public Jacobian {
public:
  explicit Chebyshev(size_t ord) : Jacobian(ord, -0.5, -0.5) {};
  virtual Chebyshev& setAlpha(double a) { throw std::out_of_range("Chebyshev polynomial; alpha is fixed."); };
  virtual Chebyshev& setBeta(double b) { throw std::out_of_range("Chebyshev polynomial; beta is fixed."); };
  virtual std::string type() { return std::string("Chebyshev"); }
};

template <typename T>
class Poly {
public:
  // Types
  typedef T poly_type;
  typedef Vector::iterator iterator;

  // Constructors
  Poly(size_t ord) : Polyn(T{ord}) {};
  Poly(size_t ord, double a, double b) : Polyn(T{ord, a, b}) {};

  // Destructors

  // Member functions
  Tensor& calcPoly();
  Tensor& calcDeriv();
  Tensor& Zeros() { return Polyn.Zeros(); }
  int order() { return Polyn.order(); };
  Poly& setOrder(size_t ord);
  double alpha() { if (Polyn.isJac()) return Polyn.alpha(); };
  double beta() { if (Polyn.isJac()) return Polyn.beta(); };
  Poly& setAlpha(double a) { if (Polyn.isJac()) Polyn.setAlpha(a); return *this; }
  Poly& setBeta(double b) { if (Polyn.isJac()) Polyn.setBeta(b); return *this; }
  Poly& setPoints(Tensor& p) { points = p; return *this; }
  std::string type() { return Polyn.type(); }

  // Iterators
  iterator pbegin() { return poly_val.begin(); }; // Polynomial iterator
  iterator pend() { return poly_val.end(); }; // Polynomial iterator
  iterator dbegin() { return deriv_val.begin(); }; // Derivative iterator
  iterator dend() { return deriv_val.end(); }; // Derivative iterator
  iterator pebegin() { return points.begin(); }; // Points evaluated iterator
  iterator peend() { return points.end(); }; // Points evaluated iterator
  iterator zbegin() { return Polyn.zbegin(); } // Zeros iterator
  iterator zend() { return Polyn.zend(); } // Zeros iterator


private:
  T Polyn;
  Vector points;
  Vector poly_val;
  Vector deriv_val;
};

template <typename T>
Tensor& Poly<T>::calcPoly() {
  if (points.empty()) throw std::out_of_range("No points set for the polynomial.");
  poly_val.resize(points.size());
  Tensor::iterator val = poly_val.begin();
  if (Polyn.isJac())
    for (Tensor::iterator it = points.begin(); it != points.end(); ++it)
      *val++ = Polyn.calcPoly(*it);
  return poly_val;
}

template <typename T>
Tensor& Poly<T>::calcDeriv() {
  if (points.empty()) throw std::out_of_range("No points set for the polynomial.");
  deriv_val.resize(points.size());
  Tensor::iterator val = deriv_val.begin();
  if (Polyn.isJac())
    for (Tensor::iterator it = points.begin(); it != points.end(); ++it)
      *val++ = Polyn.calcDeriv(*it);
  return deriv_val;
}

template <typename T>
Poly<T>& Poly<T>::setOrder(size_t ord) {
  Polyn.setOrder(ord);
  if (!poly_val.empty()) poly_val.free();
  if (!deriv_val.empty()) deriv_val.free();
  return *this; 
}

class BaseQuadType {
public:

  explicit BaseQuadType(size_t np) : Q(np) {};

  virtual Tensor& zeros(Tensor&) = 0;
  virtual Tensor& weights(Tensor&, Tensor&) = 0;
  virtual Tensor& derivm(Tensor&, Tensor&) = 0;
  virtual std::string type() = 0;
  void setQ(size_t np) { Q = np; }

protected:
  int Q;
};

class GLL : public BaseQuadType {
public:
  explicit GLL(size_t np) : BaseQuadType(np), polyj(Q-2) {};
  virtual Tensor& zeros(Tensor&);
  virtual Tensor& weights(Tensor&, Tensor&);
  virtual Tensor& derivm(Tensor&, Tensor&);
  virtual std::string type() { return std::string("Gauss-Lobatto-Legendre"); }

protected:
  Jacobian polyj;
};

class Zeros {
public:
  // Types
  typedef Tensor::iterator iterator;

  // Constructors
  Zeros(size_t np) : Q(np), zp(Q) {};

  // Member functions
  Vector& zeros() { return zp; }
  size_t size() { return Q; }
  virtual std::string type() = 0;
  virtual Zeros& setQ(size_t np) = 0;

  // Iterators
  virtual iterator zbegin() { return zp.begin(); }
  virtual iterator zend() { return zp.end(); }
  virtual iterator wbegin() = 0;
  virtual iterator wend() = 0;


protected:
  int Q;
  Vector zp;

  virtual void init() = 0;
};

template <typename T>
class Integral : public Zeros {
public:
  // Types
  typedef T quad_type;
  typedef Tensor::iterator iterator;

  // Constructor
  Integral(size_t np) : Zeros(np), Quad(np), wp(Q) { init(); };

  // Member functions
  Vector& weights() { return wp; }
  virtual std::string type() { return Quad.type(); }
  virtual Integral& setQ(size_t np) { if (np != Q) { Q = np; Quad.setQ(Q); zp.resize(Q); wp.resize(Q); init(); } return *this; }

  // Iterators
  virtual iterator zbegin() { return zp.begin(); }
  virtual iterator zend() { return zp.end(); }
  virtual iterator wbegin() { return wp.begin(); }
  virtual iterator wend() { return wp.end(); }

protected:
  T Quad;
  Vector wp;

  virtual void init() { Quad.zeros(zp); Quad.weights(zp, wp); }
};

template <typename T>
class Derivative : public Zeros {
public:
  // Types
  typedef T quad_type;
  typedef Tensor::iterator iterator;

  // Constructor
  Derivative(size_t np) : Zeros(np), Quad(np), dmp(Q) { init(); }

  // Member functions
  Matrix& matrix() { return dmp; }
  virtual std::string type() { return Quad.type(); }
  virtual Derivative& setQ(size_t np) { if (np != Q) { Q = np; Quad.setQ(Q); zp.resize(Q); dmp.resize(Q); init(); } return *this; }

  // Iterators
  virtual iterator zbegin() { return zp.begin(); }
  virtual iterator zend() { return zp.end(); }
  virtual iterator wbegin() { throw std::out_of_range("no weights in Derivative class"); }
  virtual iterator wend() { throw std::out_of_range("no weights in Derivative class"); }
  virtual iterator mbegin() { return dmp.begin(); }
  virtual iterator mend() { return dmp.end(); }


protected:
  T Quad;
  Matrix dmp;

  virtual void init() { Quad.zeros(zp); Quad.derivm(zp, dmp); }
};

template <size_t N>
class ExpBasis {
public:
  // Types
  typedef Vector::iterator iterator;

  // Constructors
  


private:

};


template <typename T, typename F>
double Integrate(Integral<T> &quad, F &func, std::initializer_list<double> ilb = {-1, 1}) {
  if (ilb.size() != 2) throw std::out_of_range("wrong number of arguments for integral limits");
  double I=0.0, J=1.0;
  Vector map(quad.size());
  auto z = quad.zbegin();
  if (!(*ilb.begin() == -1) || !((*ilb.end()-1) == 1)) {
    // Coordinate transformation (mapping qsi to x)
    double inf = *ilb.begin(), sup = *(ilb.end()-1);
    auto mi = map.begin();
    for (auto it = quad.zbegin(); it != quad.zend(); ++it)
      *mi++ = 0.5*(1 - *it)*inf + 0.5*(1 + *it)*sup;
    z = map.begin();
    J = -0.5*inf + 0.5*sup;
  }

  for (auto it = quad.wbegin(); it != quad.wend(); ++it) {
    I += (*it)*func(*z++);
  }

  return J*I;
}

template <typename T, typename F>
Vector Derivate(Derivative<T> &deriv, F &func, std::initializer_list<double> ilb = {-1, 1}) {
  Vector fd(deriv.size());
  Vector map(deriv.size());
  double sum = 0.0, J = 1.0;
  auto dm = deriv.mbegin();
  Vector z = deriv.zeros();
  auto f = fd.begin();

  if (!(*ilb.begin() == -1) || !((*ilb.end()-1) == 1)) {
    // Coordinate transformation (mapping qsi to x)
    double inf = *ilb.begin(), sup = *(ilb.end()-1);
    auto mi = map.begin();
    for (auto it = deriv.zbegin(); it != deriv.zend(); ++it)
      *mi++ = 0.5*(1 - *it)*inf + 0.5*(1 + *it)*sup;
    z = map;
    J = -0.5*inf + 0.5*sup;
  }
  
  for (size_t i = 0; i != deriv.size(); ++i) {
    for (size_t j = 0; j != deriv.size(); ++j)
      sum += (*dm++)*func(z(j));

    *f++ = (1./J)*sum;
    sum = 0.0;
  }

  return fd;
}


#endif