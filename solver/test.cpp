#include <iostream>
#include <cmath>
#include <map>
#include "Poly.h"
#include "Galerkin.h"

#define PI 3.14159265358979323846


int main() {

  ExpBasis<Modal, GLL, Line> testexp(5, 10);
  Integral<GLL> iq(10);
  auto it = iq.zeros();
  testexp.construct();
  for (size_t i = 0; i != 6; ++i) {
    for (size_t j = 0; j != 10; ++j)
      std::cout << it(j) << "\t" << testexp(i,j) << "\n";
    std::cout << "\n";
  }

  PrincFunc<Modal, GLL, Line> aa(3,5);


  return 0;
}
