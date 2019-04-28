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

class Points {
public:
  // Types
  typedef Iterator<double> iterator;

  // Constructors
  Points() {};
  Points(const Points&);
  Points(int qt) : quantity(qt), points(new double[quantity]) {};

  // Destructors
  ~Points();

  // Member functions
  size_t size() { return quantity; };
  bool empty() { return (!quantity); };
  void resize(size_t nsz);
  void free();
  iterator begin() { return Iterator<double>(points); };
  iterator end() { return Iterator<double>(points+quantity); };

  // Overloaded operators
  Points& operator=(Points&);
  double& operator[](size_t i) { if (check_range(i)) return points[i]; };

protected:
  size_t quantity = 0;
  size_t *usage = new size_t(1);
  double *points;

private:
  bool check_range(size_t);
};

class Pairs : public Points {

};

class Plane : public Points {

};

class BasePoly {
public:
  // Types
  typedef Points::iterator iterator;

  // Constructors
  explicit BasePoly(int ord) : P(ord), zeros(P) { };

  // Member functions
  size_t order() { return P; };
  virtual double calcPoly(double) = 0;
  virtual double calcDeriv(double) = 0;
  Points& Zeros() { return zeros; };

  // Iterator
  Points::iterator zbegin() { return zeros.begin(); }
  Points::iterator zend() { return zeros.end(); }

protected:  
  size_t P;
  Points zeros;
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
  virtual Points& Zeros() { return zeros; }
  bool isJac() { return hasAB; };

protected:
  double JacP(const size_t&, const double&, const double&, const double&);
  double dJacP(const size_t&, const double&, const double&, const double&);
  Points& JacPZ(const size_t&, const double&, const double&, Points&);
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
  typedef Points::iterator iterator;

  // Constructors
  Poly(int ord) : Polyn(T{ord}) {};
  Poly(int ord, double a, double b) : Polyn(T{ord, a, b}) {};

  // Destructors

  // Member functions
  Points& calcPoly();
  Points& calcDeriv();
  Points& Zeros() { return Polyn.Zeros(); }
  int order() { return Polyn.order(); };
  Poly& setOrder(size_t ord);
  double alpha() { if (Polyn.isJac()) return Polyn.alpha(); };
  double beta() { if (Polyn.isJac()) return Polyn.beta(); };
  Poly& setAlpha(double a) { if (Polyn.isJac()) Polyn.setAlpha(a); return *this; }
  Poly& setBeta(double b) { if (Polyn.isJac()) Polyn.setBeta(b); return *this; }
  Poly& setPoints(Points& p) { points = p; return *this; }
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
  Points points;
  Points poly_val;
  Points deriv_val;
};

template <typename T>
Points& Poly<T>::calcPoly() {
  if (points.empty()) throw std::out_of_range("No points set for the polynomial.");
  poly_val.resize(points.size());
  Points::iterator val = poly_val.begin();
  if (Polyn.isJac())
    for (Points::iterator it = points.begin(); it != points.end(); ++it)
      *val++ = Polyn.calcPoly(*it);
  return poly_val;
}

template <typename T>
Points& Poly<T>::calcDeriv() {
  if (points.empty()) throw std::out_of_range("No points set for the polynomial.");
  deriv_val.resize(points.size());
  Points::iterator val = deriv_val.begin();
  if (Polyn.isJac())
    for (Points::iterator it = points.begin(); it != points.end(); ++it)
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

  virtual Points& zeros(Points&) = 0;
  virtual Points& weights(Points&, Points&) = 0;
  virtual std::string type() = 0;
  void setQ(size_t np) { Q = np; }

protected:
  size_t Q;
};

class GLL : public BaseQuadType {
public:
  GLL(size_t np) : BaseQuadType(np), polyj(Q-2) {};
  virtual Points& zeros(Points&);
  virtual Points& weights(Points&, Points&);
  virtual std::string type() { return std::string("Gauss-Lobatto-Legendre"); }

private:
  Jacobian polyj;
};

template <typename T>
class Quadrature {
public:
  // Types
  typedef T quad_type;
  typedef Points::iterator iterator;

  // Constructors
  Quadrature(size_t np) : Quad(np), Q(np), zp(Q), wp(Q) { calc(); };

  // Member functions
  const Points& zeros() { return zp; }
  const Points& weights() { return wp; }
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
  Points zp;
  Points wp;

  void calc() { Quad.zeros(zp); Quad.weights(zp, wp); }
};

template <typename T, typename F>
double Integrate(Quadrature<T> &quad, F &func, double inf = -1, double sup = 1) {
  double I=0.0, J=1.0;
  Points map(quad.size());
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