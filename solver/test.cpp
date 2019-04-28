#include <iostream>
#include <cmath>
#include "../inc/Poly.h"


struct Func {
  double operator()(double x) {
    return std::sin(x);
  }
};

int main() {

  double PI = 3.14159265358979323846;

  size_t Q = 2;
  Poly<Jacobian> leg(Q, 1, 1);
  Func func;
  Quadrature<GLL> quad(Q);
  for (int i = Q; i != 9; ++i) {
    std::cout << Q << "\t" << std::fabs(Integrate(quad.setQ(Q++), func, 0, PI/2)-1.0) << std::endl;
  }

  return 0;
}
