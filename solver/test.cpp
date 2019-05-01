#include <iostream>
#include <cmath>
#include <map>
#include <type_traits>
//#include "Poly.h"
//#include "Galerkin.h"
#include "Tensor.h"

#define PI 3.14159265358979323846

Tensor<2> test(Tensor<2>& a) {
  return a;
}


int main() {


  //PrincFunc<Modal, GLL, Line> aa(3,5);
  Tensor<4> vec(5,4,3,2);
  for (int l = 0; l != 2; ++l)
    for (int k = 0; k != 3; ++k)
      for (int i = 0; i != 5; ++i)
        for (int j = 0; j != 4; ++j)
          vec(i,j,k,l);

        std::cout << std::boolalpha << vec.empty() << std::endl;


  return 0;
}
