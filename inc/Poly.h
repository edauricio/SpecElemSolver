#ifndef POLY_H
#define POLY_H

#include <iterator>
#include <string>
#include <cstddef>
#include <map>
#include <initializer_list>
#include "Tensor.h"


class BasePoly {
public:
  // Types
  typedef TensorClass::Iterator<double> iterator;

  // Constructors
  explicit BasePoly(size_t ord) : P(ord), zeros(ord) { };

  // Member functions
  size_t order() { return P; };
  virtual double calcPoly(double) = 0;
  virtual double calcDeriv(double) = 0;
  TensorClass::Tensor<1>& Zeros() { return zeros; };

  // Iterators
  TensorClass::Iterator<double> zbegin() { return zeros.begin(); }
  TensorClass::Iterator<double> zend() { return zeros.end(); }

protected:  
  int P;
  TensorClass::Tensor<1> zeros;
  bool hasAB;

};

class Jacobian : public BasePoly {
  friend class GLL;
  friend class Modal;

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
  virtual TensorClass::Tensor<1>& Zeros() { return zeros; }
  bool isJac() { return hasAB; };

protected:
  double JacP(const size_t&, const double&, const double&, const double&);
  double dJacP(const size_t&, const double&, const double&, const double&);
  TensorClass::Tensor<1>& JacPZ(const size_t&, const double&, const double&, TensorClass::Tensor<1>&);
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
  typedef TensorClass::Iterator<double> iterator;

  // Constructors
  Poly(size_t ord) : Polyn(T{ord}) {};
  Poly(size_t ord, double a, double b) : Polyn(T{ord, a, b}) {};

  // Destructors

  // Member functions
  TensorClass::Tensor<1>& calcPoly();
  TensorClass::Tensor<1>& calcDeriv();
  TensorClass::Tensor<1>& Zeros() { return Polyn.Zeros(); }
  int order() { return Polyn.order(); };
  Poly& setOrder(size_t ord);
  double alpha() { if (Polyn.isJac()) return Polyn.alpha(); };
  double beta() { if (Polyn.isJac()) return Polyn.beta(); };
  Poly& setAlpha(double a) { if (Polyn.isJac()) Polyn.setAlpha(a); return *this; }
  Poly& setBeta(double b) { if (Polyn.isJac()) Polyn.setBeta(b); return *this; }
  Poly& setPoints(TensorClass::Tensor<1>& p) { points = p; return *this; }
  std::string type() { return Polyn.type(); }

  // TensorClass::Iterators
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
  TensorClass::Tensor<1> points;
  TensorClass::Tensor<1> poly_val;
  TensorClass::Tensor<1> deriv_val;
};

template <typename T>
TensorClass::Tensor<1>& Poly<T>::calcPoly() {
  if (points.isEmpty()) throw std::out_of_range("No points set for the polynomial.");
  poly_val.resize(points.size());
  TensorClass::Iterator<double> val = poly_val.begin();
  if (Polyn.isJac())
    for (TensorClass::Iterator<double> it = points.begin(); it != points.end(); ++it)
      *val++ = Polyn.calcPoly(*it);
  return poly_val;
}

template <typename T>
TensorClass::Tensor<1>& Poly<T>::calcDeriv() {
  if (points.isEmpty()) throw std::out_of_range("No points set for the polynomial.");
  deriv_val.resize(points.size());
  TensorClass::Iterator<double> val = deriv_val.begin();
  if (Polyn.isJac())
    for (TensorClass::Iterator<double> it = points.begin(); it != points.end(); ++it)
      *val++ = Polyn.calcDeriv(*it);
  return deriv_val;
}

template <typename T>
Poly<T>& Poly<T>::setOrder(size_t ord) {
  Polyn.setOrder(ord);
  if (!poly_val.isEmpty()) poly_val.clear();
  if (!deriv_val.isEmpty()) deriv_val.clear();
  return *this; 
}

class BaseQuadType {
public:

  explicit BaseQuadType(size_t np) : Q(np) {};

  virtual TensorClass::Tensor<1>& zeros(TensorClass::Tensor<1>&) = 0;
  virtual TensorClass::Tensor<1> zeros() = 0;
  virtual TensorClass::Tensor<1>& weights(TensorClass::Tensor<1>&, TensorClass::Tensor<1>&) = 0;
  virtual TensorClass::Tensor<2>& derivm(TensorClass::Tensor<1>&, TensorClass::Tensor<2>&) = 0;
  virtual std::string type() = 0;
  void setQ(size_t np) { Q = np; }
  size_t size() { return Q; }
protected:
  int Q;
};

class GLL : public BaseQuadType {
public:
  explicit GLL(size_t np) : BaseQuadType(np), polyj(Q-2) {};
  virtual TensorClass::Tensor<1>& zeros(TensorClass::Tensor<1>&);
  virtual TensorClass::Tensor<1> zeros();
  virtual TensorClass::Tensor<1>& weights(TensorClass::Tensor<1>&, TensorClass::Tensor<1>&);
  virtual TensorClass::Tensor<2>& derivm(TensorClass::Tensor<1>&, TensorClass::Tensor<2>&);
  virtual std::string type() { return std::string("Gauss-Lobatto-Legendre"); }

protected:
  Jacobian polyj;
};

class Zeros {
public:
  // Types
  typedef TensorClass::Iterator<double> iterator;

  // Constructors
  Zeros(size_t np) : Q(np), zp(Q) {};

  // Member functions
  TensorClass::Tensor<1>& zeros() { return zp; }
  size_t size() { return Q; }
  virtual std::string type() = 0;
  virtual Zeros& setQ(size_t np) = 0;

  // TensorClass::Iterators
  virtual iterator zbegin() { return zp.begin(); }
  virtual iterator zend() { return zp.end(); }
  virtual iterator wbegin() = 0;
  virtual iterator wend() = 0;


protected:
  int Q;
  TensorClass::Tensor<1> zp;

  virtual void init() = 0;
};

template <typename T>
class Integral : public Zeros {
public:
  // Types
  typedef T quad_type;
  typedef TensorClass::Iterator<double> iterator;

  // Constructor
  Integral(size_t np) : Zeros(np), Quad(np), wp(Q) { init(); };

  // Member functions
  TensorClass::Tensor<1>& weights() { return wp; }
  virtual std::string type() { return Quad.type(); }
  virtual Integral& setQ(size_t np) { if (np != Q) { Q = np; Quad.setQ(Q); zp.resize(Q); wp.resize(Q); init(); } return *this; }

  // TensorClass::Iterators
  virtual iterator zbegin() { return zp.begin(); }
  virtual iterator zend() { return zp.end(); }
  virtual iterator wbegin() { return wp.begin(); }
  virtual iterator wend() { return wp.end(); }

protected:
  T Quad;
  TensorClass::Tensor<1> wp;

  virtual void init() { Quad.zeros(zp); Quad.weights(zp, wp); }
};

template <typename T>
class Derivative : public Zeros {
public:
  // Types
  typedef T quad_type;
  typedef TensorClass::Iterator<double> iterator;

  // Constructor
  Derivative(size_t np) : Zeros(np), Quad(np), dmp(Q,Q) { init(); }

  // Member functions
  TensorClass::Tensor<2>& matrix() { return dmp; }
  virtual std::string type() { return Quad.type(); }
  virtual Derivative& setQ(size_t np) { if (np != Q) { Q = np; Quad.setQ(Q); zp.resize(Q); dmp.resize(Q,Q); init(); } return *this; }

  // TensorClass::Iterators
  virtual iterator zbegin() { return zp.begin(); }
  virtual iterator zend() { return zp.end(); }
  virtual iterator wbegin() { throw std::out_of_range("no weights in Derivative class"); }
  virtual iterator wend() { throw std::out_of_range("no weights in Derivative class"); }
  virtual iterator mbegin() { return dmp.begin(); }
  virtual iterator mend() { return dmp.end(); }


protected:
  T Quad;
  TensorClass::Tensor<2> dmp;

  virtual void init() { Quad.zeros(zp); Quad.derivm(zp, dmp); }
};


template <typename T, typename F>
double Integrate(Integral<T> &quad, F &func, std::initializer_list<double> ilb = {-1, 1}) {
  if (ilb.size() != 2) throw std::out_of_range("wrong number of arguments for integral limits");
  double I=0.0, J=1.0;
  TensorClass::Tensor<1> map(quad.size());
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
TensorClass::Tensor<1> Derivate(Derivative<T> &deriv, F &func, std::initializer_list<double> ilb = {-1, 1}) {
  TensorClass::Tensor<1> fd(deriv.size());
  TensorClass::Tensor<1> map(deriv.size());
  double sum = 0.0, J = 1.0;
  auto dm = deriv.mbegin();
  TensorClass::Tensor<1> z = deriv.zeros();
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