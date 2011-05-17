#ifndef MATVEC_BASE_H
#define MATVEC_BASE_H

#include <iostream>
#include "blas.h"

/************************************************************
 *  Matrix & Vector storage types
************************************************************/
enum MatrixStorageType{DENSE,SYM,TRI};
enum VectorStorageType{GEN,SPARSE};

/************************************************************
 * Arithmetic Operation Types 
 ************************************************************/
enum OpType{OpAdd,   //addition
	    OpSub,   //subtraction
	    OpMul,   //multiplication
	    OpDiv,   //division
	    OpNone}; //no operation

/************************************************************
 * TypeInfo class
 *  This is used in the policy inheritance structure
 *   to reference types within various matrices and vectors
 *  Also, this will be used to access the derived class type
 *   and derived class object of a Matrix_base/Vector_base
 *   object
 ************************************************************/
template<typename T>
class TypeInfo{
public:				
  typedef T Tdata;
  typedef T Tlhs;
  typedef T Trhs;
  typedef T Tobj;
  static const OpType Op = OpNone;
};

/************************************************************
 * MatSizeInfo/VecSizeInfo class : allows for efficiently 
 *  determining the size of an arbitrary Matrix/Vector object
 *  note that this will produce a compile-time error if the
 *  template parameter is not a Matrix/Vector compatible type
 *
 * These non-specialized declarations should work for any
 *  Matrix/Vector type or Matrix_base/Vector_base type. They
 *  will be specialized below for MatrixOp/VectorOp types
 ************************************************************/
template<class T>
class MatSizeInfo{
public:
  MatSizeInfo(const T& obj) : obj_( obj.obj() ){}
  size_t nrows() const{return obj_.nrows();}
  size_t ncols() const{return obj_.ncols();}
private:
  const typename TypeInfo<T>::Tobj& obj_;
};

template<class T>
class VecSizeInfo{
public:
  VecSizeInfo(const T& obj) : obj_( obj.obj() ){}
  size_t size() const{return obj_.nrows();}
  int inc() const{return obj_.inc();}
private:
  const typename TypeInfo<T>::Tobj& obj_;
};


/************************************************************
 *  Matrix and Vector class template declarations
 *
 *   A Matrix/Vector must have a lhs(), rhs(), and obj() 
 *    methods, in order to derive from Matrix_base or 
 *    Vector_base. For a regular Matrix or Vector, lhs(), 
 *    rhs(), and obj() are itself.
 ************************************************************/
template<class T, VectorStorageType VST=GEN>
  class Vector;

template<class T, MatrixStorageType MST=DENSE>
  class Matrix;



/************************************************************
 *  TypeInfo for Matrix and Vector
 ************************************************************/
template<class T, VectorStorageType VST>
class TypeInfo<Vector<T,VST> >{
public:	
  typedef Vector<T,VST> Tobj; 	
  typedef Tobj Tlhs;
  typedef Tobj Trhs;
  typedef typename TypeInfo<T>::Tdata Tdata;
  static const OpType Op = OpNone;
};

template<class T, MatrixStorageType MST>
class TypeInfo<Matrix<T,MST> >{
public:	
  typedef Matrix<T,MST> Tobj; 
  typedef Tobj Tlhs;
  typedef Tobj Trhs;
  typedef typename TypeInfo<T>::Tdata Tdata;
  static const OpType Op = OpNone;
};



/************************************************************
 *  Scalar Class
 *   used in MatVecOp
 ************************************************************/
template<class T>
class Scalar{
 public:
  Scalar(T val) : val_(val){}
  T obj() const{return val_;}
 private:
  T val_;
};


/************************************************************
 * TypeInfo for Scalar
 ************************************************************/
template<typename T>
class TypeInfo<Scalar<T> >{
public:				
  typedef T Tdata;
  typedef T Tobj;
};



/************************************************************
 * Forward Declarations of VectorOp, MatrixOp
 *  for operator overloading in Matrix_base & Vector_base
 ************************************************************/
template<OpType Op, class Tlhs, class Trhs>
class MatrixOp;

template<OpType Op, class Tlhs, class Trhs>
class VectorOp;

/************************************************************
 * Matrix_base: 
 *   base class for Matrices
 ************************************************************/
template<class Tobj>
class Matrix_base{
public:
  static const OpType Op = TypeInfo<Tobj>::Op;
  typedef typename TypeInfo<Tobj>::Tlhs Tlhs;
  typedef typename TypeInfo<Tobj>::Trhs Trhs;
  typedef typename TypeInfo<Tobj>::Tdata Tdata;
  
  virtual const Tlhs& lhs() const = 0;
  virtual const Trhs& rhs() const = 0;
  virtual const Tobj& obj() const = 0;
  virtual Tobj& obj() = 0;

  virtual size_t nrows() const = 0;
  virtual size_t ncols() const = 0;
  
  virtual ~Matrix_base(){}
  
  //overloaded operators
  /*** operator= ***/

  //  default operator=
  Tobj& operator=(const Matrix_base& M){
    return blas_COPY(obj(),M.obj());
  }
  
  //  M = M2
  template<class T>
  Tobj& operator=(const Matrix_base<T>& M){
    return blas_COPY(obj(),M.obj());
  }
  
  //  M = ?+?
  template<class Tlhs, class Trhs>
  Tobj& operator=(const MatrixOp <OpAdd,Tlhs,Trhs>& MB){
    *this = MB.lhs();
    *this += MB.rhs();
    return obj();
  }

    //  M = ?-?
  template<class Tlhs, class Trhs>
  Tobj& operator=(const MatrixOp<OpSub,Tlhs,Trhs>& MB){
    *this = MB.lhs();
    *this -= MB.rhs();
    return obj();
  }

  //  M = M1*M2
  template<class Tlhs, class Trhs>
  Tobj& operator= (const MatrixOp<OpMul,Tlhs,Trhs>& MB){
    return blas_MM(obj(),1.0,MB.lhs(),MB.rhs(),0.0);
  }

  //  M = ?*a
  template<class T1,typename T2>
  Tobj& operator=(const MatrixOp<OpMul,Matrix_base<T1>,
		  Scalar<T2> >& MB){
    *this = MB.lhs();
    return blas_SCAL(obj(),MB.rhs());
  }
  
  /*** operator+= ***/
  //  M += ?
  template<class T>
  Tobj& operator+=(const Matrix_base<T>& MB){
    return blas_AXPY(obj(),1.0,MB.obj());
  }
  
  //  M += scalar
  Tobj& operator+=(Tdata rhs){
    return blas_AXPY(obj(),1.0,rhs);
  }
  
  //  M += ?+?
  template<class Tlhs, class Trhs>
  Tobj& operator+=(const MatrixOp<OpAdd,Tlhs,Trhs>& MB){
    *this += MB.lhs();
    *this += MB.rhs();
    return obj();
  }

  //  M += ?-?
  template<class Tlhs, class Trhs>
  Tobj& operator+=(const MatrixOp<OpSub,Tlhs,Trhs>& MB){
    *this += MB.lhs();
    *this -= MB.rhs();
    return obj();
  }

  //  M += M2*a
  template<class T1, typename T2>
    Tobj& operator+=(const MatrixOp<OpMul,Matrix_base<T1>,
		     Scalar<T2> >& MB){
    return blas_AXPY(obj(),MB.rhs(),MB.lhs());
  }

  //  M += M1*M2
  template<class Tlhs, class Trhs>
  Tobj& operator+= (const MatrixOp<OpMul,Tlhs,Trhs>& MB){
    return blas_MM(obj(),1.0,MB.lhs(),MB.rhs(),1.0);
  }

  /*** operator-= ***/
  //  M -= ?
  template<class T>
  Tobj& operator-=(const Matrix_base<T>& MB){
    return blas_AXPY(obj(),-1.0,MB.obj());
  }
  
  //  M -= scalar
  Tobj& operator-=(Tdata rhs){
    return blas_AXPY(obj(),-1.0,rhs);
  }
  
  //  M -= ?+?
  template<class Tlhs, class Trhs>
  Tobj& operator-=(const MatrixOp<OpAdd,Tlhs,Trhs>& MB){
    *this -= MB.lhs();
    *this -= MB.rhs();
    return obj();
  }

  //  M -= ?-?
  template<class Tlhs, class Trhs>
  Tobj& operator-=(const MatrixOp<OpSub,Tlhs,Trhs>& MB){
    *this -= MB.lhs();
    *this += MB.rhs();
    return obj();
  }

  //  M -= M2*a
  template<class T1,typename T2>
  Tobj& operator-=(const MatrixOp<OpMul,Matrix_base<T1>,
		   Scalar<T2> >& MB){
    
    return blas_AXPY(obj(),-MB.rhs(),MB.lhs());
  }

  //  M -= M1*M2
  template<class Tlhs, class Trhs>
  Tobj& operator-= (const MatrixOp<OpMul,Tlhs,Trhs>& MB){
    return blas_MM(obj(),-1.0,MB.lhs(),MB.rhs(),1.0);
  }

  /*** operator*= ***/
  //  M *= a
  template<class T>
  Tobj& operator*=(const T& rhs){
    return blas_SCAL(obj(),Tdata(rhs) );
  }

  /*** operator/= ***/
  //  M /= a
  template<class T>
  Tobj& operator/=(const T& rhs){
    return blas_SCAL(obj(),Tdata(1.0/rhs) );
  }
};

/************************************************************
 * Vector_base: 
 *   base class for Vectors
 ************************************************************/

template<typename Tobj>
class Vector_base{
public:
  static const OpType Op = TypeInfo<Tobj>::Op;
  typedef typename TypeInfo<Tobj>::Tlhs Tlhs;
  typedef typename TypeInfo<Tobj>::Trhs Trhs;
  typedef typename TypeInfo<Tobj>::Tdata Tdata;
  
  virtual const Tlhs& lhs() const = 0;
  virtual const Trhs& rhs() const = 0;
  virtual const Tobj& obj() const = 0;
  virtual Tobj& obj() = 0;

  virtual size_t size() const = 0;
  
  virtual ~Vector_base(){}

  //overloaded operators
  /*** operator= ***/

  //  default operator=
  Tobj& operator=(const Vector_base& M){
    return blas_COPY(obj(),M.obj());
  }
  
  //  V = V2
  template<class T>
  Tobj& operator=(const Vector_base<T>& V){
    return blas_COPY(obj(),V.obj());
  }
  
  //  V = ?+?
  template<class Tlhs, class Trhs>
  Tobj& operator=(const VectorOp<OpAdd,Tlhs,Trhs>& VB){
    *this = VB.lhs();
    *this += VB.rhs();
    return obj();
  }

  //  V = ?-?
  template<class Tlhs, class Trhs>
  Tobj& operator=(const VectorOp<OpSub,Tlhs,Trhs>& VB){
    *this = VB.lhs();
    *this -= VB.rhs();
    return obj();
  }

  //  V = ?*a
  template<class T1, typename T2>
  Tobj& operator=(const VectorOp<OpMul,Vector_base<T1>,
		  Scalar<T2> >& VB){
    *this = VB.lhs();
    return blas_SCAL(obj(),VB.rhs());
  }

  //  V = M*V2
  template<class Tlhs,class Trhs>
    Tobj& operator=(const VectorOp<OpMul,Matrix_base<Tlhs>, 
		    Vector_base<Trhs> >& VB){
    return blas_MV(obj(),1.0,VB.lhs(),VB.rhs(),0.0);
  }

  //  V = V2*M
  template<class Tlhs,class Trhs>
    Tobj& operator=(const VectorOp<OpMul,Vector_base<Tlhs>, 
		    Matrix_base<Trhs> >& VB){
    return blas_MV(obj(),1.0,VB.rhs().Transpose(),VB.lhs(),0.0);
  }
  
  /*** operator+= ***/

  //  V += ?
  template<class TT>
  Tobj& operator+=(const Vector_base<TT>& VB){
    return blas_AXPY(obj(),1.0,VB.obj());
  }
  
  //  V += scalar
  Tobj& operator+=(Tdata rhs){
    return blas_AXPY(obj(),1.0,rhs);
  }

  //  V += ?+?
  template<class Tlhs, class Trhs>
  Tobj& operator+=(const VectorOp<OpAdd,Tlhs,Trhs>& VB){
    *this += VB.lhs();
    *this += VB.rhs();
    return obj();
  }

  //  V += ?-?
  template<class Tlhs, class Trhs>
  Tobj& operator+=(const VectorOp<OpSub,Tlhs,Trhs>& VB){
    *this += VB.lhs();
    *this -= VB.rhs();
    return obj();
  }

  //  V += V2*a
  template<class T1, typename T2>
  Tobj& operator+=(const VectorOp<OpMul,Vector_base<T1>,
		   Scalar<T2> >& VB){
    return blas_AXPY(obj(),VB.rhs(),VB.lhs());
  }

  //  V = M*V2
  template<class Tlhs,class Trhs>
    Tobj& operator+=(const VectorOp<OpMul,Matrix_base<Tlhs>, 
		    Vector_base<Trhs> >& VB){
    return blas_MV(obj(),1.0,VB.lhs(),VB.rhs(),1.0);
  }

  //  V = V2*M
  template<class Tlhs,class Trhs>
    Tobj& operator+=(const VectorOp<OpMul,Vector_base<Tlhs>, 
		    Matrix_base<Trhs> >& VB){
    return blas_MV(obj(),1.0,VB.rhs().Transpose(),VB.lhs(),1.0);
  }

  /*** operator-= ***/
  //  V -= ?
  template<class TT>
  Tobj& operator-=(const Vector_base<TT>& VB){
    return blas_AXPY(obj(),-1.0,VB.obj());
  }
  
  //  V -= scalar
  Tobj& operator-=(Tdata rhs){
    return blas_AXPY(obj(),-1.0,rhs);
  }

  //  V -= ?+?
  template<class Tlhs, class Trhs>
  Tobj& operator-=(const VectorOp<OpAdd,Tlhs,Trhs>& VB){
    *this -= VB.lhs();
    *this -= VB.rhs();
    return obj();
  }

  //  V -= ?-?
  template<class Tlhs, class Trhs>
  Tobj& operator-=(const VectorOp<OpSub,Tlhs,Trhs>& VB){
    *this -= VB.lhs();
    *this += VB.rhs();
    return obj();
  }

  //  V -= V2*a
  template<class T1, typename T2>
  Tobj& operator-=(const VectorOp<OpMul,Vector_base<T1>,
		   Scalar<T2> >& VB){
    return blas_AXPY(obj(),-VB.rhs(),VB.lhs());
  }

  //  V = M*V2
  template<class Tlhs,class Trhs>
    Tobj& operator-=(const VectorOp<OpMul,Matrix_base<Tlhs>, 
		    Vector_base<Trhs> >& VB){
    return blas_MV(obj(),-1.0,VB.lhs(),VB.rhs(),1.0);
  }

  //  V = V2*M
  template<class Tlhs,class Trhs>
    Tobj& operator-=(const VectorOp<OpMul,Vector_base<Tlhs>, 
		    Matrix_base<Trhs> >& VB){
    return blas_MV(obj(),-1.0,VB.rhs().Transpose(),VB.lhs(),1.0);
  }

  /*** operator*= ***/
  //  V *= ?
  template<class T>
  Tobj& operator*=(const T& rhs){
    return blas_SCAL(obj(),Tdata(rhs) );
  }

  /*** operator/= ***/
  //  V /= ?
  template<class T>
  Tobj& operator/=(const T& rhs){
    return blas_SCAL(obj(),Tdata(1.0/rhs) );
  }
};



/************************************************************
 * TypeInfo for Matrix_base and Vector_base
 ************************************************************/
template<typename T>
class TypeInfo<Matrix_base<T> >{
public:				
  typedef typename TypeInfo<T>::Tdata Tdata;
  typedef typename TypeInfo<T>::Tlhs Tlhs;
  typedef typename TypeInfo<T>::Trhs Trhs;
  typedef typename TypeInfo<T>::Tobj Tobj;
  static const OpType Op = TypeInfo<T>::Op;
};


template<typename T>
class TypeInfo<Vector_base<T> >{
public:				
  typedef typename TypeInfo<T>::Tlhs Tlhs;
  typedef typename TypeInfo<T>::Trhs Trhs;
  typedef typename TypeInfo<T>::Tdata Tdata;
  typedef typename TypeInfo<T>::Tobj Tobj;
  static const OpType Op = TypeInfo<T>::Op;
};


/************************************************************
 * MatrixOp/VectorOp : encapsulates arithmetic for Matrices
 *   and Vectors, with need for allocating large temporary
 *   objects.  This uses the TypeInfo class liberally to make
 *   sure that lhs and rhs are pointing to the correct
 *   objects, namely, of the highest derived type.
 ************************************************************/
template<OpType Op, class Tlhs, class Trhs>
class MatrixOp : public Matrix_base<MatrixOp<Op,Tlhs,Trhs> >{
 private:
  typedef typename TypeInfo<Tlhs>::Tobj Tlhs_obj;
  typedef typename TypeInfo<Trhs>::Tobj Trhs_obj;
  const Tlhs_obj& lhs_;
  const Trhs_obj& rhs_;
 public:
  MatrixOp(const Matrix_base<MatrixOp<Op,Tlhs,Trhs> >& MB) 
    : lhs_(MB.lhs()), rhs_(MB.rhs()){}
    
  MatrixOp(const Tlhs& lhs, const Trhs& rhs) 
    : lhs_( lhs.obj() ), rhs_( rhs.obj() ){}

  const Tlhs_obj& lhs() const{return lhs_;}

  const Trhs_obj& rhs() const{return rhs_;}

  const MatrixOp& obj() const{return *this;}
  MatrixOp& obj(){return *this;}

  size_t nrows() const{
    return MatSizeInfo<MatrixOp<Op,Tlhs,Trhs> >(*this).nrows();}
  
  size_t ncols() const{
    return MatSizeInfo<MatrixOp<Op,Tlhs,Trhs> >(*this).ncols();}
};



template<OpType Op, class Tlhs, class Trhs>
class VectorOp : public Vector_base<VectorOp<Op,Tlhs,Trhs> >{
 private:
  typedef typename TypeInfo<Tlhs>::Tobj Tlhs_obj;
  typedef typename TypeInfo<Trhs>::Tobj Trhs_obj;
  const Tlhs_obj& lhs_;
  const Trhs_obj& rhs_;
 public:
  VectorOp(const Vector_base<VectorOp<Op,Tlhs,Trhs> >& VB) 
    : lhs_(VB.lhs()), rhs_(VB.rhs()){}

  VectorOp(const Tlhs& lhs, const Trhs& rhs) 
    : lhs_( lhs.obj() ), rhs_( rhs.obj() ){}

  const Tlhs_obj& lhs() const{return lhs_;}

  const Trhs_obj& rhs() const{return rhs_;}

  const VectorOp& obj() const{return *this;}
  VectorOp& obj(){return *this;}

  size_t size() const{
    return VecSizeInfo<VectorOp<Op,Tlhs,Trhs> >(*this).size();}
};


/************************************************************
 * TypeInfo for MatrixOp and VectorOp
 ************************************************************/
template<OpType Op_, typename Tlhs_, typename Trhs_>
class TypeInfo<MatrixOp<Op_,Tlhs_,Trhs_> >{
public:				
  typedef MatrixOp<Op_,Tlhs_,Trhs_> Tobj;
  typedef typename TypeInfo<Tlhs_>::Tobj Tlhs;
  typedef typename TypeInfo<Trhs_>::Tobj Trhs;
  typedef typename TypeInfo<Tlhs>::Tdata Tdata;
  static const OpType Op = Op_;
};

template<OpType Op_, typename Tlhs_, typename Trhs_>
class TypeInfo<VectorOp<Op_,Tlhs_,Trhs_> >{
public:	
  typedef VectorOp<Op_,Tlhs_,Trhs_> Tobj;
  typedef typename TypeInfo<Tlhs_>::Tobj Tlhs;
  typedef typename TypeInfo<Trhs_>::Tobj Trhs;
  typedef typename TypeInfo<Tlhs>::Tdata Tdata;
  static const OpType Op = Op_;
};

/************************************************************
 * SizeInfo for MatrixOp and VectorOp
 ************************************************************/
//General MatSizeInfo
template<OpType Op, class Tlhs, class Trhs>
class MatSizeInfo<MatrixOp<Op,Tlhs,Trhs> >{
public:
  MatSizeInfo(const MatrixOp<Op,Tlhs,Trhs>& MO) 
    : obj_(MO.lhs()){}
  size_t nrows() const{return obj_.nrows();}
  size_t ncols() const{return obj_.ncols();}
private:
  const typename TypeInfo<Tlhs>::Tobj& obj_;
};

//Matrix-Matrix multiplication
template<class Trhs,class Tlhs>
class MatSizeInfo<MatrixOp<OpMul,Matrix_base<Tlhs>,Matrix_base<Trhs> > >{
public:
  MatSizeInfo(const MatrixOp<OpMul,Matrix_base<Tlhs>,Matrix_base<Trhs> >& MO) 
    : lhs_(MO.lhs()) , rhs_(MO.rhs()){}
  size_t nrows() const{return lhs_.nrows();}
  size_t ncols() const{return rhs_.ncols();}
private:
  const typename TypeInfo<Tlhs>::Tobj& lhs_;
  const typename TypeInfo<Trhs>::Tobj& rhs_;
};

//Matrix-Scalar operation
template<OpType Op, class Tlhs>
class MatSizeInfo<MatrixOp<Op,Matrix_base<Tlhs>,
			   typename TypeInfo<Tlhs>::Tdata> >{
public:
  MatSizeInfo(const MatrixOp<OpMul,Matrix_base<Tlhs>,
	      typename TypeInfo<Tlhs>::Tdata>& MO) : obj_(MO.lhs()){}
  size_t nrows() const{return obj_.nrows();}
  size_t ncols() const{return obj_.ncols();}
private:
  const typename TypeInfo<Tlhs>::Tobj& obj_;
};

//General VecSizeInfo
template<OpType Op, class Tlhs, class Trhs>
class VecSizeInfo<VectorOp<Op,Tlhs,Trhs> >{
public:
  VecSizeInfo(const VectorOp<Op,Tlhs,Trhs>& MO) : obj_(MO.lhs()){}
  size_t size() const{return obj_.size();}
private:
  const typename TypeInfo<Tlhs>::Tobj& obj_;
};

//Matrix-Vector multiplication
template<class Trhs,class Tlhs>
class VecSizeInfo<VectorOp<OpMul,Matrix_base<Tlhs>,Vector_base<Trhs> > >{
public:
  VecSizeInfo(const VectorOp<OpMul,Matrix_base<Tlhs>,Vector_base<Trhs> >& VO) 
    : obj_(VO.lhs()){}
  size_t size() const{return obj_.nrows();}
private:
  const typename TypeInfo<Tlhs>::Tobj& obj_;
};

//Vector-Matrix multiplication
template<class Trhs,class Tlhs>
class VecSizeInfo<VectorOp<OpMul,Vector_base<Tlhs>,Matrix_base<Trhs> > >{
public:
  VecSizeInfo(const VectorOp<OpMul,Vector_base<Tlhs>,Matrix_base<Trhs> >& VO) 
    : obj_(VO.lhs()){}
  size_t size() const{return obj_.ncols();}
private:
  const typename TypeInfo<Tlhs>::Tobj& obj_;
};

//Vector-Scalar operation
template<OpType Op, class Tlhs>
class MatSizeInfo<MatrixOp<Op,Vector_base<Tlhs>,
			   typename TypeInfo<Tlhs>::Tdata> >{
public:
  MatSizeInfo(const MatrixOp<OpMul,Tlhs,typename TypeInfo<Tlhs>::Tdata>& MO) 
  : obj_(MO.lhs()){}
  size_t nrows() const{return obj_.nrows();}
  size_t ncols() const{return obj_.ncols();}
private:
  const typename TypeInfo<Tlhs>::Tobj& obj_;
};


/************************************************************
 * General Matrix operator overloading
 ************************************************************/
/*** Matrix Addition ***/
// M + M
template<typename Tlhs, typename Trhs>
inline MatrixOp< OpAdd, Matrix_base<Tlhs>, Matrix_base<Trhs> >
operator+(const Matrix_base<Tlhs>& lhs, const Matrix_base<Trhs>& rhs){
  return MatrixOp< OpAdd, Matrix_base<Tlhs>, Matrix_base<Trhs> >(lhs,rhs);
}

/*** Matrix Subtraction ***/
// M - M
template<typename Tlhs, typename Trhs>
inline MatrixOp< OpSub, Matrix_base<Tlhs>, Matrix_base<Trhs> >
operator-(const Matrix_base<Tlhs>& lhs, const Matrix_base<Trhs>& rhs){
  return MatrixOp< OpSub, Matrix_base<Tlhs>, Matrix_base<Trhs> >(lhs,rhs);
}

/*** Matrix Multiplication ***/
// M * M
template<typename Tlhs, typename Trhs>
inline MatrixOp< OpMul, Matrix_base<Tlhs>, Matrix_base<Trhs> >
operator*(const Matrix_base<Tlhs>& lhs, const Matrix_base<Trhs>& rhs){
  return MatrixOp< OpMul, Matrix_base<Tlhs>, Matrix_base<Trhs> >(lhs,rhs);
}

// M * a
template<typename T>
inline MatrixOp< OpMul, Matrix_base<T>, 
		 Scalar<typename Matrix_base<T>::Tdata> >
operator*(const Matrix_base<T>& M, const typename Matrix_base<T>::Tdata& a){
  typedef Scalar<typename Matrix_base<T>::Tdata> DT;
  return MatrixOp<OpMul,Matrix_base<T>,DT>(M,DT(a));
}

// a * M
template<typename T>
inline MatrixOp< OpMul, Matrix_base<T>, 
		 Scalar<typename Matrix_base<T>::Tdata> >
operator*(const typename Matrix_base<T>::Tdata& a, const Matrix_base<T>& M){
  typedef Scalar<typename Matrix_base<T>::Tdata> DT;
  return MatrixOp<OpMul,Matrix_base<T>,DT>(M,DT(a));
}

/*** Matrix Division ***/
// M / a
template<typename T>
inline MatrixOp< OpMul, Matrix_base<T>, 
		 Scalar<typename Matrix_base<T>::Tdata> >
operator/(const Matrix_base<T>& M, const typename Matrix_base<T>::Tdata& a){
  typedef Scalar<typename Matrix_base<T>::Tdata> DT;
  return MatrixOp<OpMul,Matrix_base<T>,DT>(M,DT(1.0/a));
}

/*** Unary Negation ***/
// -M
template<typename T>
inline MatrixOp< OpMul, Matrix_base<T>, 
		 Scalar<typename Matrix_base<T>::Tdata> >
operator-(const Matrix_base<T>& M){
  typedef Scalar<typename Matrix_base<T>::Tdata> DT;
  return MatrixOp<OpMul,Matrix_base<T>,DT>(M,DT(-1));
}


/************************************************************
 * General Vector operator overloading
 ************************************************************/
/*** Vector Addition ***/
// V + V
template<typename Tlhs, typename Trhs>
inline VectorOp< OpAdd, Vector_base<Tlhs>, Vector_base<Trhs> >
operator+(const Vector_base<Tlhs>& lhs, const Vector_base<Trhs>& rhs){
  return VectorOp< OpAdd, Vector_base<Tlhs>, Vector_base<Trhs> >(lhs,rhs);
}

/*** Vector Subtraction ***/
// V - V
template<typename Tlhs, typename Trhs>
inline VectorOp< OpSub, Vector_base<Tlhs>, Vector_base<Trhs> >
operator-(const Vector_base<Tlhs>& lhs, const Vector_base<Trhs>& rhs){
  return VectorOp< OpSub, Vector_base<Tlhs>, Vector_base<Trhs> >(lhs,rhs);
}

/*** Vector Multiplication ***/
// M * V
template<typename Tlhs, typename Trhs>
inline VectorOp< OpMul, Matrix_base<Tlhs>, Vector_base<Trhs> >
operator*(const Matrix_base<Tlhs>& lhs, const Vector_base<Trhs>& rhs){
  return VectorOp< OpMul, Matrix_base<Tlhs>, Vector_base<Trhs> >(lhs,rhs);
}

// V * M
template<typename Tlhs, typename Trhs>
inline VectorOp< OpMul, Vector_base<Tlhs>, Matrix_base<Trhs> >
operator*(const Vector_base<Tlhs>& lhs, const Matrix_base<Trhs>& rhs){
  return VectorOp< OpMul, Vector_base<Tlhs>, Matrix_base<Trhs> >(lhs,rhs);
}

// V * a
template<typename T>
inline VectorOp< OpMul, Vector_base<T>, 
		 Scalar<typename Vector_base<T>::Tdata> >
operator*(const Vector_base<T>& V, const typename Vector_base<T>::Tdata& a){
  typedef Scalar<typename Vector_base<T>::Tdata> DT;
  return VectorOp<OpMul,Vector_base<T>,DT>(V,DT(a));
}

// a * V
template<typename T>
inline VectorOp< OpMul, Vector_base<T>, 
		 Scalar<typename Vector_base<T>::Tdata> >
operator*(const typename Vector_base<T>::Tdata& a, const Vector_base<T>& V){
  typedef Scalar<typename Vector_base<T>::Tdata> DT;
  return VectorOp<OpMul,Vector_base<T>,DT>(V,DT(a));
}

/*** Vector Division ***/
// V / a
template<typename T>
inline VectorOp< OpMul, Vector_base<T>, 
		 Scalar<typename Vector_base<T>::Tdata> >
operator/(const Vector_base<T>& V, const typename Vector_base<T>::Tdata& a){
  typedef Scalar<typename Vector_base<T>::Tdata> DT;
  return VectorOp<OpMul,Vector_base<T>,DT>(V,DT(1.0/a));
}

/*** Unary Negation ***/
// -V
template<typename T>
inline VectorOp< OpMul, Vector_base<T>, 
		 Scalar<typename Vector_base<T>::Tdata> >
operator-(const Vector_base<T>& V){
  typedef Scalar<typename Vector_base<T>::Tdata> DT;
  return VectorOp<OpMul,Vector_base<T>,DT>(V,DT(-1));
}

#endif /* MATVEC_BASE_H */
