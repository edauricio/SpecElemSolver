#ifndef POLY_H
#define POLY_H

#include <iterator>
#include <string>
#include <cstddef>
#include <map>

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

template <size_t N>
class Tensor {
public:
  // Types
  typedef Iterator<double> iterator;

  // Constructors
  Tensor() {};
  Tensor(const Tensor&);
  Tensor(int qt) : quantity(qt), points(new double[quantity]) {};

  // Destructors
  ~Tensor();

  // Member functions
  size_t size() { return quantity; };
  bool empty() { return (!quantity); };
  void resize(size_t nsz);
  void free();
  iterator begin() { return Iterator<double>(points); };
  iterator end() { return Iterator<double>(points+quantity); };

  // Overloaded operators
  Tensor& operator=(Tensor&);
  double& operator[](size_t i) { if (check_range(i)) return points[i]; };

protected:
  size_t quantity = 0;
  size_t *usage = new size_t(1);
  double *points;

private:
  bool check_range(size_t);
};

template <size_t N>
Tensor<N>::Tensor(const Tensor<N>& p) : quantity(p.quantity), usage(p.usage), points(p.points) {
  (*usage)++;
}

template <size_t N>
Tensor<N>::~Tensor() {
  (*usage)--;
  if (!(*usage)) {
    if (quantity) delete[] points;
    delete usage;
  }
}

template <size_t N>
Tensor<N>& Tensor<N>::operator=(Tensor<N>& rhs) {
  // Increase counter for RHS
  (*(rhs.usage))++;

  // Decrease counter for LHS (usage operation orders is safe for self-assignment)
  (*usage)--;
  if (!(*usage)) {
    if (quantity) delete[] points;
    delete usage;
  }

  // Adjust LHS pointers accordingly
  quantity = rhs.quantity;
  usage = rhs.usage;
  points = rhs.points;
}

template <size_t N>
void Tensor<N>::resize(size_t nsz) { 
  if ((*usage) > 1) throw std::out_of_range("Can't resize a shared Tensor object.");
  if (quantity) delete[] points; 
  quantity = nsz; 
  points = new double[nsz]; 
}

template <size_t N>
void Tensor<N>::free() {
  if ((*usage) > 1) throw std::out_of_range("Can't free a shared Tensor object.");
  if (!quantity) return;
  delete[] points;
  quantity = 0;
}

template <size_t N>
bool Tensor<N>::check_range(size_t i) {
  if (i < quantity) return true;
  throw std::out_of_range("index out of points range");
}

class BasePoly {
public:
  // Types
  typedef Tensor<1>::iterator iterator;

  // Constructors
  explicit BasePoly(int ord) : P(ord), zeros(P) { };

  // Member functions
  size_t order() { return P; };
  virtual double calcPoly(double) = 0;
  virtual double calcDeriv(double) = 0;
  Tensor<1>& Zeros() { return zeros; };

  // Iterator
  Tensor<1>::iterator zbegin() { return zeros.begin(); }
  Tensor<1>::iterator zend() { return zeros.end(); }

protected:  
  size_t P;
  Tensor<1> zeros;
  bool hasAB;

};

class Jacobian : public BasePoly {
  friend class GLL;

public:
  // Constructors
  explicit Jacobian(int ord, double a = 1., double b = 1.) : BasePoly(ord), Alpha(a), Beta(b) { hasAB = true; JacPZ(P, Alpha, Beta, zeros); };

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
  virtual Tensor<1>& Zeros() { return zeros; }
  bool isJac() { return hasAB; };

protected:
  double JacP(const size_t&, const double&, const double&, const double&);
  double dJacP(const size_t&, const double&, const double&, const double&);
  Tensor<1>& JacPZ(const size_t&, const double&, const double&, Tensor<1>&);
  void JacPZ(const size_t&, const double&, const double&, iterator, iterator);
  double Alpha, Beta;

private:
  void clear_n_zeros(bool = false);
};

class Legendre : public Jacobian {
public:
  explicit Legendre(int ord) : Jacobian(ord, 0, 0) {};
  virtual Legendre& setAlpha(double a) { throw std::out_of_range("Legendre polynomial; alpha is fixed."); };
  virtual Legendre& setBeta(double b) { throw std::out_of_range("Legendre polynomial; beta is fixed."); };
  virtual std::string type() { return std::string("Legendre"); }
};

class Chebyshev : public Jacobian {
public:
  explicit Chebyshev(int ord) : Jacobian(ord, -0.5, -0.5) {};
  virtual Chebyshev& setAlpha(double a) { throw std::out_of_range("Chebyshev polynomial; alpha is fixed."); };
  virtual Chebyshev& setBeta(double b) { throw std::out_of_range("Chebyshev polynomial; beta is fixed."); };
  virtual std::string type() { return std::string("Chebyshev"); }
};

template <typename T>
class Poly {
public:
  // Types
  typedef T poly_type;
  typedef Iterator<double> iterator;

  // Constructors
  Poly(int ord) : Polyn(T{ord}) {};
  Poly(int ord, double a, double b) : Polyn(T{ord, a, b}) {};

  // Destructors

  // Member functions
  Tensor<1>& calcPoly();
  Tensor<1>& calcDeriv();
  Tensor<1>& Zeros() { return Polyn.Zeros(); }
  int order() { return Polyn.order(); };
  Poly& setOrder(size_t ord);
  double alpha() { if (Polyn.isJac()) return Polyn.alpha(); };
  double beta() { if (Polyn.isJac()) return Polyn.beta(); };
  Poly& setAlpha(double a) { if (Polyn.isJac()) Polyn.setAlpha(a); return *this; }
  Poly& setBeta(double b) { if (Polyn.isJac()) Polyn.setBeta(b); return *this; }
  template <size_t N>
  Poly& setPoints(Tensor<N>& p) { points = p; return *this; }
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
  template <size_t N>
  Tensor<N> points;
  Tensor<1> poly_val;
  Tensor<1> deriv_val;
};

template <typename T>
Tensor<1>& Poly<T>::calcPoly() {
  if (points.empty()) throw std::out_of_range("No points set for the polynomial.");
  poly_val.resize(points.size());
  Tensor<1>::iterator val = poly_val.begin();
  if (Polyn.isJac())
    for (Tensor<1>::iterator it = points.begin(); it != points.end(); ++it)
      *val++ = Polyn.calcPoly(*it);
  return poly_val;
}

template <typename T>
Tensor<1>& Poly<T>::calcDeriv() {
  if (points.empty()) throw std::out_of_range("No points set for the polynomial.");
  deriv_val.resize(points.size());
  Tensor<1>::iterator val = deriv_val.begin();
  if (Polyn.isJac())
    for (Tensor<1>::iterator it = points.begin(); it != points.end(); ++it)
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

  BaseQuadType(size_t np) : Q(np) {};

  virtual Tensor<1>& zeros(Tensor<1>&) = 0;
  virtual Tensor<1>& weights(Tensor<1>&, Tensor<1>&) = 0;
  virtual std::string type() = 0;
  void setQ(size_t np) { Q = np; }

protected:
  size_t Q;
};

class GLL : public BaseQuadType {
public:
  GLL(size_t np) : BaseQuadType(np), polyj(Q-2) {};
  virtual Tensor<1>& zeros(Tensor<1>&);
  virtual Tensor<1>& weights(Tensor<1>&, Tensor<1>&);
  virtual std::string type() { return std::string("Gauss-Lobatto-Legendre"); }

private:
  Jacobian polyj;
};

template <typename T>
class Quadrature {
public:
  // Types
  typedef T quad_type;
  typedef Iterator<double> iterator;

  // Constructors
  Quadrature(size_t np) : Quad(np), Q(np), zp(Q), wp(Q) { calc(); };

  // Member functions
  const Tensor<1>& zeros() { return zp; }
  const Tensor<1>& weights() { return wp; }
  size_t size() { return Q; }
  std::string type() { return Quad.type(); }
  Quadrature& setQ(size_t np) { if (np != Q) { Q = np; Quad.setQ(Q); zp.resize(Q); wp.resize(Q); calc(); } return *this; }

  // Iterators
  iterator zbegin() { return zp.begin(); }
  iterator zend() { return zp.end(); }
  iterator wbegin() { return wp.begin(); }
  iterator wend() { return wp.end(); }

private:
  T Quad;
  size_t Q;
  Tensor<1> zp;
  Tensor<1> wp;
  Tensor<2> dm;

  void calc() { Quad.zeros(zp); Quad.weights(zp, wp); }
};

template <typename T, typename F>
double Integrate(Quadrature<T> &quad, F &func, double inf = -1, double sup = 1) {
  double I=0.0, J=1.0;
  Tensor<1> map(quad.size());
  auto z = quad.zbegin();
  if (!(inf == -1) || !(sup == 1)) {
    // Coordinate transformation (mapping qsi to x)
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

#endif