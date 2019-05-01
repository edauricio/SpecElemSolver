#ifndef TENSOR_H
#define TENSOR_H

#include <type_traits>
#include <cstddef>
#include <stdexcept>
#include <iterator>

namespace TensorClass {

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
  int size(size_t = 0) const; // Return size of i-th dimension
  bool isEmpty() const { return (!nval); }
  bool isSquare() const { if (N != 2) throw std::out_of_range("isSquare() available only for matrix (rank 2 tensor)"); return (dim[0] == dim[1]); }
  void clear(); // Free memory used by Tensor and restore it to "null state"

  template <typename... Szs>
  Tensor& resize(Szs...);

  // Element access operations
  template <typename... As>
  double& operator()(As...);

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

  /* Member functions to check compatibility of arguments type */
  template <typename C>
  void type_check(C);

  template <typename C, typename... Rest>
  void type_check(C, Rest...);
  /* ------- END OF ARGS TYPE CHECK ------- */

  /* Member functions to fill the dim array on initialization of object */
  template <typename D>
  void fill_dim(size_t*, D);

  template <typename D, typename... Rest>
  void fill_dim(size_t*, D, Rest...);
  /* ------- END OF FUNCS TO FILL DIM ARRAY ------- */


  /* Member functions to get the mapped 1D index requested by Multidim. indices of operator() */
  template <typename I>
  void get_ind(I);

  template <typename I, typename J>
  void get_ind(I, J);

  template <typename I, typename J, typename... Rest>
  void get_ind(I, J, Rest...);

  template <typename I>
  void get_mult(I);

  template <typename I, typename... Rest>
  void get_mult(I, Rest...);
  /* ------ END OF FUNCS TO GET MAPPED 1D INDEX REQUESTED BY OPERATOR() ------ */

  /* Member functions to check if resize is really needed when resize() is called */
  template <typename S>
  bool check_resize(size_t*, S);

  template <typename S, typename... Rest>
  bool check_resize(size_t*, S, Rest...);
  /* ------ END OF FUNCS TO CHECK IF RESIZE() SHOULD REALLY MESS WITH *values ----- */


};

template <size_t N>
template <typename... Ns>
Tensor<N>::Tensor(Ns... ns) {
  if (sizeof...(ns) != N) throw std::out_of_range("number of arguments for Tensor constructor and Tensor dimensions do not match");
  type_check(ns...);

  // Check-pass: Tensor is now going to be constructed, so create usage storage
  usage = new size_t(1);

  // Filling dim array with values for each dimension of tensor
  dim = new size_t[N]();
  fill_dim(dim, ns...);

  // Allocating storage for tensor values array
  nval = 1;
  for (size_t i = 0; i != N; ++i) {
    nval *= dim[i];
  }
  values = new double[nval]();
}

template <size_t N>
bool Tensor<N>::check_range(size_t i) {
  if (i < nval) return true;
  throw std::out_of_range("element access denied: index out of range");
}

template <size_t N>
int Tensor<N>::size(size_t i) const {
  if (!nval) throw std::out_of_range("empty Tensor has no size()");
  if (i > N-1) throw std::out_of_range("argument to Tensor size() exceeded tensor rank"); 
  return dim[i]; 
}

template <size_t N>
void Tensor<N>::clear() {
  if (usage && *usage > 1) throw std::out_of_range("can't clear() shared Tensor object");
  if (values) delete[] values;
  if (dim) delete[] dim;
  if (usage) delete usage;

  nval = 0;
  usage = nullptr;
  dim = nullptr;
  values = nullptr;
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
  //if (!d1) throw std::out_of_range("Tensor can't have a dimension with 0 elements");
  *p = d1;
}

template <size_t N>
template <typename D, typename... Rest>
void Tensor<N>::fill_dim(size_t *p, D d1, Rest... drest) {
  //if (!d1) throw std::out_of_range("Tensor can't have a dimension with 0 elements");
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
double& Tensor<N>::operator()(As... is) {
  if (!nval) throw std::out_of_range("trying to access empty Tensor");
  // Checking indices number and types
  if (sizeof...(is) != N) throw std::out_of_range("element access denied: incorrect number of indices for tensor of rank N");
  type_check(is...);

  /*Getting the mapped 1D index
  * Mapping is according to:

  * \left[\sum_{p=3}^{N}\left(i_p\prod_{k=1}^{p-1}x_k\right)\right] + x_2i_1 + i_2

  * where
  * i_p are the arguments for the element access operator, that is, 
  the multidimensional indices (i.e. i_p = i, j, k, ..., for p = 1, 2, 3, ...)

  * x_k are the number of elements in each dimension k
  * *NOTE: the above formula is NOT in C-style index notation (i.e. indices start at 1 there, not 0).
  */
  elind = 0;
  get_ind(is...);
  // Check if it is in range of values array
  check_range(elind);
  return *(values+elind);
}

template <size_t N>
template <typename I>
void Tensor<N>::get_ind(I i) {
  elind += i;
}

template <size_t N>
template <typename I, typename J>
void Tensor<N>::get_ind(I i, J j) {
  elind += i*dim[1] + j;
}

template <size_t N>
template <typename I, typename J, typename... Rest>
void Tensor<N>::get_ind(I i, J j, Rest... r){
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

// Get the multiplicative part of the mapping
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

template <size_t N>
template <typename... Szs>
Tensor<N>& Tensor<N>::resize(Szs... sz) {
  if (usage && *usage > 1) throw std::out_of_range("can't resize a shared tensor");
  if (sizeof...(sz) != N) throw std::out_of_range("number of arguments to resize() differs from rank tensor");
  type_check(sz...);
  if (usage) {    
    if (check_resize(dim, sz...)) {
      if (values) delete[] values;

      fill_dim(dim, sz...);

      nval = 1;
      for (size_t i = 0; i != N; ++i)
        nval *= dim[i];

      values = new double[nval]();
    }
  } else {
    usage = new size_t(1);
    dim = new size_t[N];
    fill_dim(dim, sz...);

    nval = 1;
      for (size_t i = 0; i != N; ++i)
        nval *= dim[i];

      values = new double[nval]();
  }
}

template <size_t N>
template <typename S>
bool Tensor<N>::check_resize(size_t *d, S sz) {
  if (*d != sz) return true;
  return false;
}

template <size_t N>
template <typename S, typename... Rest>
bool Tensor<N>::check_resize(size_t *d, S sz, Rest... r) {
  if (*d != sz) return true;
  if (check_resize(d+1, r...)) return true;
  return false;
}

} // end of TensorClass namespace

#endif