#include <iostream>
#include <cmath>
#include <map>
#include <type_traits>
#include "Tensor.h"
//#include "Poly.h"
//#include "Galerkin.h"


#define PI 3.14159265358979323846

Tensor<2> test(Tensor<2>& a) {
  a.resize(2, 2);
  return a;
}


int main() {


  //PrincFunc<Modal, GLL, Line> aa(3,5);
  Tensor<2> vec(5,4);

      for (int i = 0; i != 5; ++i)
        for (int j = 0; j != 4; ++j)
          vec(i,j);
        vec.resize(10,10);

  Tensor<2> vec2;
  vec2.resize(2,2);
  vec2(1,1);
  std::cout << vec2.size(0) << "\t" << vec2.size(1) << "\n";
  vec2.resize(5,5);
  std::cout << vec2.size(0) << "\t" << vec2.size(1) << "\n";

  
  return 0;
}
