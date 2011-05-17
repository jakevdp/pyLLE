#include "MatVec.h"

/************************************************************
 * Matrix Definitions
 ************************************************************/

template<class T>
Matrix<T,DENSE>::Matrix(size_t nrows/*=0*/,size_t ncols/*=0*/,T initval/*=0*/)
  : arr_(0),nrows_(nrows),ncols_(ncols),mem_allocated(false), trans_(false){
  if(nrows_*ncols_ == 0){
    nrows_ = 0;
    ncols_ = 0;
  }
  else{
    arr_ = new T[nrows_ * ncols_];
    mem_allocated = true;
    SetAllTo(initval);
  }
}


template<class T>
Matrix<T,DENSE>::Matrix(size_t nrows,size_t ncols,T *arr,bool trans)
  : arr_(arr),nrows_(nrows),ncols_(ncols),mem_allocated(false),trans_(trans){
  if(trans){
    nrows_ = ncols;
    ncols_ = nrows;
  }
}

template<class T>
Matrix<T,DENSE>::Matrix(const Matrix<T,DENSE>& M)
  : arr_(M.arr_),nrows_(M.nrows_),ncols_(M.ncols_),
    mem_allocated(false),trans_(M.trans_){}

template<class T>
Matrix<T,DENSE>::~Matrix(){
  if(mem_allocated)
    delete [] arr_;
}

template<class T>
void Matrix<T,DENSE>::deep_copy(const Matrix &M){
  if(arr_ != M.arr_){
    reallocate(M.nrows(),M.ncols());
    *this = M;
  }
  trans_ = M.trans_;
}

template<class T>
void Matrix<T,DENSE>::viewof(const Matrix &M){
  if(arr_ != M.arr_){
    reallocate(0,0);
    arr_ = M.arr_;
    mem_allocated = false;
  }
  ncols_ = M.ncols_;
  nrows_ = M.nrows_;
  trans_ = M.trans_;
}

template<class T>
void Matrix<T,DENSE>::reallocate(size_t nrows,size_t ncols,T initval/* = 0*/){
  trans_ = false;
  if( mem_allocated && (nrows_*ncols_ == nrows*ncols) ){
    nrows_ = nrows;
    ncols_ = ncols;
    SetAllTo(initval);
  }else if(nrows*ncols == 0){
    nrows_ = 0;
    ncols_ = 0;
    arr_ = 0;
    mem_allocated = false;
  }else{
    nrows_ = nrows;
    ncols_ = ncols;
    if(mem_allocated)
      delete [] arr_;
    arr_ = new T[nrows_ * ncols_];
    mem_allocated = true;
    SetAllTo(initval);
  }
}

template<class T>
void Matrix<T,DENSE>::swap(Matrix &M){
  T* arr_tmp = arr_;
  arr_ = M.arr_;
  M.arr_ = arr_tmp;

  size_t nrows_tmp = nrows_;
  nrows_ = M.nrows_;
  M.nrows_ = nrows_tmp;

  size_t ncols_tmp = ncols_;
  ncols_ = M.ncols_;
  M.ncols_ = ncols_tmp;
  
  bool mem_allocated_tmp = mem_allocated;
  mem_allocated = M.mem_allocated;
  M.mem_allocated = mem_allocated_tmp;

  bool trans_tmp = trans_;
  trans_ = M.trans_;
  M.trans_ = trans_tmp;
}

template<class T>
T& Matrix<T,DENSE>::operator()(size_t i, size_t j){
  if(i>=nrows() || j>=ncols())
    throw IndexError();
  else if(trans_)
    return arr_[i*nrows_+j];
  else
    return arr_[j*nrows_+i];
}

template<class T>
const T& Matrix<T,DENSE>::operator()(size_t i,size_t j) const{
  if(i>=nrows() || j>=ncols())
    throw IndexError();
  else if(trans_)
    return arr_[i*nrows_+j];
  else
    return arr_[j*nrows_+i];
}

template<class T>
Vector<T,GEN> Matrix<T,DENSE>::row(size_t i){
  if(i>=nrows())
    throw IndexError();
  else if(trans_)
    return Vector<T,GEN>(nrows_,arr_+i*nrows_,1);
  else
    return Vector<T,GEN>(ncols_,arr_+i,nrows_);
}

template<class T>
const Vector<T,GEN> Matrix<T,DENSE>::row(size_t i) const{
  if(i>=nrows())
    throw IndexError();
  else if(trans_)
    return Vector<T,GEN>(nrows_,arr_+i*nrows_,1);
  else
    return Vector<T,GEN>(ncols_,arr_+i,nrows_);
}

template<class T>
Vector<T,GEN> Matrix<T,DENSE>::col(size_t j){
  if(j>=ncols())
    throw IndexError();
  else if(trans_)
    return Vector<T,GEN>(ncols_,arr_+j,nrows_);
  else
    return Vector<T,GEN>(nrows_,arr_+j*nrows_,1);
}

template<class T>
const Vector<T,GEN> Matrix<T,DENSE>::col(size_t j) const{
  if(j>=ncols())
    throw IndexError();
  else if(trans_)
    return Vector<T,GEN>(ncols_,arr_+j,nrows_);
  else
    return Vector<T,GEN>(nrows_,arr_+j*nrows_,1);
}

template<class T>
Vector<T,GEN> Matrix<T,DENSE>::diag(){
  return Vector<T,GEN>( (nrows_<ncols_)?nrows_:ncols_, arr_, nrows_+1);
}

template<class T>
const Vector<T,GEN> Matrix<T,DENSE>::diag() const{
  return Vector<T,GEN>( (nrows_<ncols_)?nrows_:ncols_, arr_, nrows_+1);
}


template<class T>
void Matrix<T,DENSE>::SetAllTo(T val){
  //call blas_COPY here?
  for(size_t i=0;i<nrows();i++)
    for(size_t j=0;j<ncols();j++){
      operator()(i,j) = val;
    }
}

template<class T>
T Matrix<T,DENSE>::SumElements() const{
  T S = 0;
  for(size_t i=0; i<nrows_; i++)
    for(size_t j=0; j<ncols_; j++)
      S += arr_[j*nrows_ + i];
  return S;
}

/************************************************************
 * Vector Definitions
 ************************************************************/
template<class T>
Vector<T,GEN>::Vector(size_t size, T initval)
  : arr_(NULL), inc_(1), size_(size), mem_allocated(false){
  if(size>0){
    arr_ = new T[size];
    mem_allocated = true;
  }
  SetAllTo(initval);
}

template<class T>
Vector<T,GEN>::Vector(size_t size,T *arr,int inc) 
  : arr_(arr), inc_(inc), size_(size), mem_allocated(false){}

template<class T>
Vector<T,GEN>::Vector(const Vector<T,GEN>& V)
  : arr_(V.arr_), inc_(V.inc_), size_(V.size_), mem_allocated(false){}

template<class T>
Vector<T,GEN>::~Vector(){
  if (mem_allocated){
    delete [] arr_;
  }
}


template<class T>
void Vector<T,GEN>::deep_copy(const Vector<T,GEN> &V){
  if(arr_ != V.arr_){
    reallocate(V.size());
    // call blas_COPY here?
    for(size_t i=0;i<size_;i++)
      (*this)(i) = V(i);
  }
}

template<class T>
void Vector<T,GEN>::viewof(const Vector<T,GEN> &V){
  if(arr_ != V.arr_){
    reallocate(0);
    arr_ = V.arr_;
    mem_allocated = false;
  }
  size_ = V.size_;
  inc_ = V.inc_;
}

template<class T>
void Vector<T,GEN>::swap(size_t i, size_t j){
  T temp = operator()(i);
  operator()(i) = operator()(j);
  operator()(j) = temp;
}

template<class T>
void Vector<T,GEN>::reallocate(size_t size,T initval){
  //deallocate memory
  if (mem_allocated && (size_ = size) ){
    size_ = size;
  }else if(size == 0){
    size_ = 0;
    arr_ = 0;
    mem_allocated = false;
  }else{
    size_ = size;
    if(mem_allocated)
      delete [] arr_;
    arr_ = new T[size_];
    mem_allocated = true;
    SetAllTo(initval);
  }
}

template<class T>
T& Vector<T,GEN>::operator()(size_t i){
  if(i >= size_)
    throw IndexError();
  else
    return arr_[i * inc_];
}
 
template<class T> 
const T& Vector<T,GEN>::operator()(size_t i) const{
  if(i >= size_)
    throw IndexError();
  else
    return arr_[i * inc_];
}

template<class T>
Vector<T,GEN> Vector<T,GEN>::SubVector(size_t start,size_t end){
  if(start > size_ || end > size_)
    throw IndexError();
  return Vector<T,GEN>(end-start, arr_ + inc_*start, inc_);
}

template<class T>
void Vector<T,GEN>::SetAllTo(T val){
  //call blas_COPY here?
  for(size_t i=0;i<size_;i++)
    arr_[i*inc_] = val;
}

template<class T>
T Vector<T,GEN>::SumElements() const{
  T S = 0;
  for(size_t i =0; i<size_; i++)
    S += arr_[i*inc_];
  return S;
}


//explicitly declare double DENSE matrices and double/int GEN vectors: 
// this is to prevent needing all the above
// template functions to be defined in the header
// (prevents code bloat...)
template class Vector<double,GEN>;
template class Vector<int,GEN>;
template class Matrix<double,DENSE>;
template class Matrix<int,DENSE>;
