#ifndef ARGSORT_H
#define ARGSORT_H

#include "MatVec.h"
#include "MatVecExcept.h"
#include <limits>
#include <vector>
#include <algorithm>

//structure to help argsort
template <class T>
struct LT_Indices{
  const Vector<T>* x_ptr;
  LT_Indices(const Vector<T> &x) {x_ptr = &x;}
  bool operator()(int i1, int i2) const {return ((*x_ptr)(i1) < (*x_ptr)(i2));}
};


//function argsort
// given a vector x,
//  returns a vector of indices to sort x in increasing order
template <class T>
void argsort(const Vector<T>& x , Vector<int>& indices)
{
  unsigned int N = x.size();
  
  if(indices.size() != N)
    indices.reallocate(N,0);
  
  for(unsigned int i=0;i<N;i++)
    indices(i)=i;

  std::sort(indices.arr(), indices.arr()+N, LT_Indices<T>(x));
}

template<class T>
void argsort_these(const Vector<T>& x , Vector<int>& indices)
{
  unsigned int N = indices.size();

  std::sort(indices.arr(), indices.arr()+N, LT_Indices<T>(x));
}

template <class T>
void argsort(Vector<T>& x , Vector<int>& indices , int k){
  int N = x.size();
  if(k<0 || k==N){
    argsort(x,indices);
  }else if(k>N){
    throw MatVecError("argsort : k must be less than N");
  }else{
    std::vector<double> x_min( k,std::numeric_limits<double>::max() );
    indices.reallocate(k);
    int index;
    //first go through the first k values
    for(int i=0;i<N;i++){
      index = std::lower_bound(x_min.begin(), x_min.end(), x(i)) - x_min.begin();
      if(index>=k)
	continue;

      for(int j=k-1;j>index;j--){
	indices(j) = indices(j-1);
	x_min[j] = x_min[j-1];
      }
      x_min[index] = x(i);
      indices(index) = i;
    }
  }
}

#endif //ARGSORT_H
