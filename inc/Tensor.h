#ifndef TENSOR_H
#define TENSOR_H

#include <type_traits>
#include <cstddef>
#include <stdexcept>
#include <iostream>
#include <iterator>

template <typename T>
class Iterator : public std::iterator<std::random_access_iterator_tag, T> {
public:
  // Single Constructor
  Iterator(T* obj) : it(obj) {};

  // Operator overloading
  T& operator*() { return *it; };
  Iterator operator+(size_t s) { return this->it + s; }
  Iterator& operator++() { it += 1; return *this; };
  Iterator operator++(int) { it += 1; return this->it - 1; };
  std::ptrdiff_t operator-(const Iterator& it2) { return it - it2.it; }
  Iterator operator-(int s) { return it-s; }
  Iterator& operator--() { it -= 1; return *this; };
  Iterator operator--(int) { it -= 1; return this->it + 1; };
  bool operator==(const Iterator& it2) { return (this->it == it2.it); };
  bool operator!=(const Iterator& it2) { return (this->it != it2.it); };

private:
  T* it;
};

template <size_t N>
class Tensor {
public:
  // Types
  typedef Iterator<double> iterator;

  // Constructors
  Tensor() {};

  template <typename... Ns>
  Tensor(Ns...);

  Tensor(const Tensor&); // Copy constructor
  template <size_t R>
  Tensor(const Tensor<R>&); // Template version of "copy constructor" (not really a copy constructor since types differ);
                            // Only for exception throwing, so as to avoid a confused compile-time error due to the variadic template constructor.

  Tensor(Tensor&&); // Move constructor

  // Destructor
  ~Tensor();

  // Member functions
  int rank() const { return N; }
  bool empty() const { return (!nval); }

  // Element access operations
  template <typename... As>
  double &operator()(As...);

  // Overloaded operators
  Tensor& operator=(const Tensor&); // Copy-assignment operator
  Tensor& operator=(Tensor&&); // Move-assignment operator

  // Iterators
  iterator begin() { return iterator(values); }
  iterator end() { return iterator(values+nval); }

private:
  size_t nval = 0;
  size_t elind = 0;
  size_t *usage = nullptr;
  size_t *dim = nullptr;
  double *values = nullptr;

  bool check_range(size_t); // Function to check index range for element access

  /* Member functions to check compatibility of constructor arguments type */
  template <typename C>
  void type_check(C);

  template <typename C, typename... Rest>
  void type_check(C, Rest...);
  /* ------- END OF C'TOR ARGS TYPE CHECK ------- */

  /* Member functions to fill the dim array on initialization of object */
  template <typename D>
  void fill_dim(size_t*, D);

  template <typename D, typename... Rest>
  void fill_dim(size_t*, D, Rest...);
  /* ------- END OF FUNCS TO FILL DIM ARRAY ------- */


  /* Member functions to check type compatibility of operator() arguments */
  template <typename I>
  void ind_check(I);

  template <typename I, typename... Rest>
  void ind_check(I, Rest...);
  /* ------ END OF FUNCS TO CHECK OPERATOR() ARGS ------ */

  /* Member functions to get the mapped 1D index requested by Multidim. indices of operator() */
  template <typename I>
  void get_ind(I);

  template <typename I>
  void get_ind(I, I);

  template <typename I, typename... Rest>
  void get_ind(I, I, Rest...);

  template <typename I>
  void get_mult(I);

  template <typename I, typename... Rest>
  void get_mult(I, Rest...);
  /* ------ END OF FUNCS TO GET MAPPED 1D INDEX REQUESTED BY OPERATOR() ------ */
};

template <size_t N>
template <typename... Ns>
Tensor<N>::Tensor(Ns... ns) {
  if (sizeof...(ns) != N) throw std::out_of_range("number of arguments for Tensor constructor and Tensor dimensions do not match");
  type_check(ns...);

  // Check-pass: Tensor is now going to be constructed, so create usage storage
  usage = new size_t(1);

  // Filling dim array with values for each dimension of tensor
  dim = new size_t[N];
  fill_dim(dim, ns...);

  // Allocating storage for tensor values array
  nval = 1;
  for (size_t i = 0; i != N; ++i) {
    nval *= dim[i];
  }
  values = new double[nval];
}

template <size_t N>
bool Tensor<N>::check_range(size_t i) {
  std::cout << i << "\n";
  if (i < nval) return true;
  throw std::out_of_range("element access denied: index out of range");
}

template <size_t N>
template <typename C>
void Tensor<N>::type_check(C c) { 
  if(!std::is_integral<C>::value) throw std::out_of_range("argument(s) of Tensor constructor must be of integral type");
}

template <size_t N>
template <typename C, typename... Rest>
void Tensor<N>::type_check(C c, Rest... r) {
  if(!std::is_integral<C>::value) throw std::out_of_range("argument(s) of Tensor constructor must be of integral type");
  type_check(r...);
}

template <size_t N>
template <typename D>
void Tensor<N>::fill_dim(size_t *p, D d1) {
  if (!d1) throw std::out_of_range("Tensor can't have a dimension with 0 elements");
  *p = d1;
}

template <size_t N>
template <typename D, typename... Rest>
void Tensor<N>::fill_dim(size_t *p, D d1, Rest... drest) {
  if (!d1) throw std::out_of_range("Tensor can't have a dimension with 0 elements");
  *p = d1;
  fill_dim(p+1, drest...);
}

/* Destructor definition */
template <size_t N>
Tensor<N>::~Tensor() {
  if (usage) (*usage)--;

  if (usage && *usage == 0) {
    if (values) delete[] values;
    if (dim) delete[] dim;
    delete usage;
  }
}

template <size_t N>
template <size_t R>
Tensor<N>::Tensor(const Tensor<R>& f) {
  throw std::out_of_range("constructing Tensor from another with different dimension");
}

/* Copy-constructor definition */
template <size_t N>
Tensor<N>::Tensor(const Tensor& f) : nval(f.nval), usage(f.usage), dim(f.dim), values(f.values) {
  if (usage) (*usage)++;
}

/* Move-constructor definition */
template <size_t N>
Tensor<N>::Tensor(Tensor&& fr) : nval(fr.nval), usage(fr.usage), dim(fr.dim), values(fr.values) {
  fr.nval = 0;
  fr.usage = nullptr;
  fr.dim = nullptr;
  fr.values = nullptr;
}

/* Copy-assignment operator */
template <size_t N>
Tensor<N>& Tensor<N>::operator=(const Tensor& rhs) {
  // Check if RHS is empty, otherwise increment usage counter (copy-constructor phase)
  if (rhs.usage) (*rhs.usage)++;

  // Decrement usage counter for LHS object and free memory if counter goes to zero (destructor phase)
  if (usage) (*usage)--;

  if (usage && *usage == 0) {
    if (values) delete[] values;
    if (dim) delete[] dim;
    delete usage;
  }

  nval = rhs.nval;
  usage = rhs.usage;
  dim = rhs.dim;
  values = rhs.values;
}

/* Move-assignment operator */
template <size_t N>
Tensor<N>& Tensor<N>::operator=(Tensor&& rhs) {
  // Decrement usage counter for LHS object and free memory if counter goes to zero (destructor phase)
  if (usage) (*usage)--;

  if (usage && *usage == 0) {
    if (values) delete[] values;
    if (dim) delete[] dim;
    delete usage;
  }

  // Adjust LHS pointers and vars
  nval = rhs.nval;
  usage = rhs.usage;
  dim = rhs.dim;
  values = rhs.values;

  // Leave RHS in a "null state" after move
  rhs.nval = 0;
  rhs.usage = nullptr;
  rhs.dim = nullptr;
  rhs.values = nullptr;
}

template <size_t N>
template <typename... As>
double &Tensor<N>::operator()(As... is) {
  // Checking indices number and types
  if (sizeof...(is) != N) throw std::out_of_range("element access denied: incorrect number of indices for tensor of rank N");
  ind_check(is...);

  // Getting the mapped 1D index
  get_ind(is...);
  // Check if it is in range of values array
  check_range(elind);
  size_t temp = elind;
  elind = 0;
  return *(values+temp);
}

template <size_t N>
template <typename I>
void Tensor<N>::ind_check(I i) {
  if (!std::is_convertible<I, size_t>::value) throw std::out_of_range("element access denied: non-integral type passed as index");
}

template <size_t N>
template <typename I, typename... Rest>
void Tensor<N>::ind_check(I i, Rest... is) {
  if (!std::is_convertible<I, size_t>::value) throw std::out_of_range("element access denied: non-integral type passed as index");
  ind_check(is...);
}

template <size_t N>
template <typename I>
void Tensor<N>::get_ind(I i) {
  elind += i;
}

template <size_t N>
template <typename I>
void Tensor<N>::get_ind(I i, I j) {
  elind += i*dim[1] + j;
}

template <size_t N>
template <typename I, typename... Rest>
void Tensor<N>::get_ind(I i, I j, Rest... r){
  get_mult(r...);
  get_ind(i, j);
}

template <size_t N>
template <typename I>
void Tensor<N>::get_mult(I i) {
  size_t temp = 1;
  for (size_t i = 0; i != N-1; ++i)
    temp *= dim[i];
  temp *= i;
  elind += temp;
}

template <size_t N>
template <typename I, typename... Rest>
void Tensor<N>::get_mult(I i, Rest... r) {
  size_t temp = 1;
  for (size_t i = 0; i != N-(sizeof...(r)+1); ++i)
    temp *= dim[i];
  temp *= i;
  elind += temp;
  get_mult(r...);
}

#endif