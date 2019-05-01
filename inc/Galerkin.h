#ifndef GALERKIN_H
#define GALERKIN_H

#include "Tensor.h"
#include "Poly.h"
#include "Fem.h"

class BasePrincFType {
public:
  // Types


  // Constructors
  BasePrincFType(size_t ord) : P(ord) {};

  // Destructors


  // Member functions
  size_t size() { return P; }
  void setP(size_t ord) { P = ord; }
  virtual BasePrincFType& construct(TensorClass::Tensor<1>, TensorClass::Tensor<2>&) = 0;

  // Overloaded operators


protected:
  int P;

};

class Modal : public BasePrincFType {
public:
  // Types
  typedef TensorClass::Iterator<double>::iterator iterator;

  // Constructors
  Modal(size_t ord) : BasePrincFType(ord), polyj(ord, 1, 1) {};

  // Member functions
  virtual Modal& construct(TensorClass::Tensor<1>, TensorClass::Tensor<2>&);

private:
  Jacobian polyj;

};


/* Class to generate principal functions to be used to construct an expansion basis
Based on:
class T - Type of principal function (Modal/Nodal)
class Q - Quadrature zeros to be used to evaluate the polynomials of the principal functions
          (Gauss-Lobatto-Legendre, etc)
class E - Type of element for which the expansion basis to be constructed
          (Quad, Tri, Hexa, Prism, etc)

*/
template <class T, class Q, class E>
class PrincFunc {
public:
  // Types
  typedef T princ_func_type;
  typedef Q quad_type;
  typedef E element_type;

  // Constructors
  PrincFunc(size_t ord, size_t zs) : Princ(ord), Quad(zs) {};


  // Member functions


private:
  T Princ;
  Q Quad;
  E Ele;
  typename E::PrincFuncStruct Data;
};


/* Class to generate an expansion basis. Constructed on top of a set of principal functions, class PrincFunc.
class T - Type of principal function (Modal/Nodal)
class Q - Quadrature zeros to be used to evaluate the polynomials of the principal functions
          (Gauss-Lobatto-Legendre, etc)
class E - Type of element for which the expansion basis to be constructed
          (Quad, Tri, Hexa, Prism, etc)
*/
template <class T, class Q, class E>
class ExpBasis {
public:
  // Types
  typedef TensorClass::Iterator<double>::iterator iterator;
  typedef T exp_type;
  typedef Q quad_type;
  typedef E element_type;

  // Constructors
  ExpBasis(size_t ord, size_t zeros) : exp(ord), quad(zeros), em(ord+1,zeros) {};

  // Member functions
  ExpBasis& setP(size_t ord) { if (ord != exp.size()) { exp.setP(ord); em.resize(ord+1, quad.size()); } }
  ExpBasis& setQ(size_t zeros) { if (zeros != quad.size()) { quad.setQ(zeros); em.resize(exp.size()+1, zeros); } }
  ExpBasis& reset(size_t ord, size_t zeros) { if ((zeros != quad.size()) || (ord != exp.size())) { exp.setP(ord); quad.setQ(zeros); em.resize(ord+1, zeros); } }
  ExpBasis& construct() { exp.construct(quad.zeros(), em); }
  //ExpBasis& construct(size_t ord, size_t zeros) { exp.setP(ord); quad.setQ(zeros); em.resize(ord, zeros); exp.construct(quad.zeros, em); }

  // Overloaded operators
  double& operator()(size_t i, size_t j) { return em(i,j); }

private:
  T exp;
  Q quad;
  E elem;
  TensorClass::Tensor<2> em;
};

#endif