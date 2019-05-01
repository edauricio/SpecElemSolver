#include <cmath>
#include <iostream>
#include <stdexcept>
#include "Tensor.h"
#include "Poly.h"

using namespace TensorClass;

double Jacobian::JacP(const size_t& n, const double& alpha, const double& beta, const double& x) {
  if (n < 0 ) return 0.0;
  else if (n == 0) return 1.0;
  else if (n == 1) return 0.5*(alpha - beta + (alpha + beta + 2.0)*x);
  else if (n > 1) {
    double an1 = 2*((n-1.)+1.)*((n-1) + alpha + beta + 1.0)*(2.*(n-1.) + alpha + beta);
    double an2 = (2.*(n-1.) + alpha + beta + 1.0)*(pow(alpha,2) - pow(beta,2));
    double an3 = (2.*(n-1.) + alpha + beta)*(2.*(n-1.) + alpha + beta + 1.0)*(2.*(n-1.) + alpha + beta + 2.0);
    double an4 = 2.*((n-1.) + alpha)*((n-1.) + beta)*(2.*(n-1.) + alpha + beta + 2.0);
    return ((an2 + an3*x)*JacP(n-1, alpha, beta, x) - an4*JacP(n-2, alpha, beta, x))/an1;
  }
}

double Jacobian::dJacP(const size_t& n, const double& alpha, const double& beta, const double& x) {
  return 0.5*(n + alpha + beta + 1.0)*JacP(n-1, alpha+1., beta+1., x);
}

Tensor<1>& Jacobian::JacPZ(const size_t &n, const double &alpha , const double &beta, Tensor<1> &zeros) {
  double r=0., s, delta, tol = 1e-9;
  int i;
  for (int k = 0; k != n; ++k) {
    r = -cos(((2.*k+1.)/(2.*n))*3.14159265358979323846);
    if (k > 0) r = (r + zeros(k-1))*0.5;
    do {
      for (s = 0.0, i = 0; i != k; ++i) s += 1./(r - zeros(i));
      delta = -JacP(n, alpha, beta, r)/(dJacP(n, alpha, beta, r) - s*JacP(n, alpha, beta, r));
      r += delta;
      if (fabs(delta) < tol) break;
    } while(true);
    zeros(k) = r;
  }
  return zeros;
}

void Jacobian::JacPZ(const size_t &n, const double &alpha, const double &beta, iterator b, iterator e) {
  double r, s, delta, tol = 0.000001;
  if ((e-b) != n) throw std::out_of_range("iterator range to JacPZ must have at least the same number of elements as the order of polynomial");
  for (int k = 0; k != n; ++k) {
    r = -std::cos(((2.*k+1.)/(2.*n))*3.14159265358979323846);
    if (k > 0) r = (r + *(b+(k-1)))*0.5;
    do {
      s=0.0;
      for (int i = 0; i != k; ++i) s += 1./(r - *(b+i));
      delta = -JacP(n, alpha, beta, r)/(dJacP(n, alpha, beta, r) - s*JacP(n, alpha, beta, r));
      r += delta;
      if (std::fabs(delta) < tol) break;
    } while(true);
    *(b+k) = r;
  }
}

void Jacobian::clear_n_zeros(bool ordChange) {
  if (ordChange) zeros.resize(P);
  JacPZ(P, Alpha, Beta, zeros);
}

Tensor<1>& GLL::zeros(Tensor<1>& zp) {
  zp(0) = -1.0; zp(Q-1) = 1.0;
  polyj.reset(Q-2, 1, 1);
  size_t i = 1;
  for (auto it = polyj.zbegin(); it != polyj.zend(); ++it) zp(i++) = *it;
  return zp;
}

Tensor<1> GLL::zeros() {
  Tensor<1> zp(Q);
  zp(0) = -1.0; zp(Q-1) = 1.0;
  polyj.reset(Q-2, 1, 1);
  size_t i = 1;
  for (auto it = polyj.zbegin(); it != polyj.zend(); ++it) zp(i++) = *it;
  return zp;
}

Tensor<1>& GLL::weights(Tensor<1>& zp, Tensor<1> &wp) {
  auto iz = zp.begin();
  for (auto iw = wp.begin(); iw != wp.end(); ++iw) {
    *iw = 2./(Q*(Q-1)*std::pow(polyj.JacP(Q-1, 0, 0, *iz++), 2));
  }
  return wp;
}

Tensor<2>& GLL::derivm(Tensor<1>& zp, Tensor<2>& dmp) {
  dmp(0,0) = -Q*(Q-1)*0.25;
  for (size_t i = 0; i != dmp.size(0); ++i) {
    for (size_t j = 0; j != dmp.size(1); ++j) {
      if (i != j)
        dmp(i,j) = (polyj.JacP(Q-1, 0, 0, zp(i))/polyj.JacP(Q-1, 0, 0, zp(j)))*(1./(zp(i) - zp(j)));
    }
  }
  for (size_t i = 1; i != dmp.size(0)-1; ++i) {
    dmp(i,i) = 0.0;
  }

  dmp(dmp.size(0)-1,dmp.size(1)-1) = Q*(Q-1)*0.25;

  return dmp;
}