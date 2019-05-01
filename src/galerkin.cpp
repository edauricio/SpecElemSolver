#include "Galerkin.h"

using namespace TensorClass;

Modal& Modal::construct(Tensor<1> zp, Tensor<2>& pm) {
  if (pm.size(0) != P+1) throw std::out_of_range("size of expansion basis matrix and P doest not match");
  for (size_t i = 0; i != zp.size(); ++i)
    pm(0,i) = 0.5*(1. - zp(i));

  for (size_t p = 1; p != P; ++p)
    for (size_t i = 0; i != zp.size(); ++i)
      pm(p,i) = 0.5*(1. - zp(i))*0.5*(1. + zp(i))*polyj.JacP(p-1, 1, 1, zp(i));

  for (size_t i = 0; i != zp.size(); ++i)
    pm(P,i) = 0.5*(1.+zp(i));

  return *this;
}