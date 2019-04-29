#include <iostream>
#include <cmath>
#include "Poly.h"


// struct Func {
//   double operator()(double x) {
//     return std::sin(x);
//   }
// };

double func(double x) {
  return std::pow(x, 7);
}

int main() {

  double PI = 3.14159265358979323846;

  Derivative<GLL> deriv(8);
  Vector res = Derivate(deriv, func, {2, 10});
  auto z = deriv.zbegin();
  std::cout << "Zeros\tNum.\tExact\n";
  for (auto it = res.begin(); it != res.end(); ++it) std::cout << *z << "\t" << *it << "\t" << 7*pow(*z++,6) << "\n";


  return 0;
}
