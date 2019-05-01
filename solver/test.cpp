#include <iostream>
#include <cmath>
#include <map>
#include <type_traits>
#include "Tensor.h"
#include "Poly.h"
#include "Galerkin.h"

using namespace TensorClass;


#define PI 3.14159265358979323846

double func(double x) {
  return pow(x, 7);
}


int main() {

  int Q = 20;
  Derivative<GLL> dev(Q);
  Tensor<1> zeros = dev.zeros();
  Tensor<1> fe = Derivate(dev, func, {2, 10});
  auto ir = zeros.begin();
  auto it = fe.begin();
  for (int i = 0; i != Q; ++i)
    std::cout << -(1- *ir++) + 5*(1+*ir) << "\t" << *it++ << "\n";


  
  return 0;
}
