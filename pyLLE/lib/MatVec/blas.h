#ifndef BLAS_H
#define BLAS_H

#include "MatVecExcept.h"

/************************************************************
 high-level wrapper templates for BLAS routines
************************************************************/

//NRM2: vector/matrix 2-norm : sqrt( sum(elements^2) )
template<class Tobj>
double blas_NRM2(const Tobj&){throw NotImplementedError("blas_NRM2");}

//SCAL: vector/matrix scaling : X *= a
template<class Tlhs, class Trhs>
Tlhs& blas_SCAL(Tlhs&, Trhs){throw NotImplementedError("blas_SCAL");}

//DOT: vector dot product = V1*V2
template<class Tlhs, class Trhs>
double blas_DOT(const Tlhs&, const Trhs&){
  throw NotImplementedError("blas_DOT");}

//COPY: vector/matrix copy : V1 = V2
template<class Tlhs, class Trhs>
Tlhs& blas_COPY(Tlhs&, const Trhs&){throw NotImplementedError("blas_COPY");}

//AXPY: vector addition : y += a*x
template<class TY, class Ta, class TX>
TY& blas_AXPY(TY&, Ta, const TX&){throw NotImplementedError("blas_AXPY");}

//MV: matrix-vector product : y = alpha*A*x + beta*y
template<class TY, class Talpha, class TA, class TX, class Tbeta>
TY& blas_MV(TY&, Talpha, const TA&, const TX&, Tbeta){
  throw NotImplementedError("blas_MV");}

//MM: matrix-matrix product : C = alpha*A*B + beta*C
template<class TC, class Talpha, class TA, class TB, class Tbeta>
TC& blas_MM(TC&, Talpha, const TA&, const TB&, Tbeta){
  throw NotImplementedError("blas_MM");}

//RK: matrix rank-k update: C = alpha*A^T*A + beta*C
template<class TC, class Talpha, class TA, class Tbeta>
TC& blas_RK(TC&, Talpha, const TA&, Tbeta){
  throw NotImplementedError("blas_RK");}


/************************************************************
 BLAS routines and low-level wrappers
************************************************************/

/******** Level-1 routines ********/

//dscal : vector scaling
extern "C" void dscal_(const int *N, const double *DA, 
		       double *DX, const int *INCX);

inline void dscal(const int& N, const double& DA,
		  double* DX, const int& INCX){
  dscal_(&N, &DA, DX, &INCX);
  //std::cout << " - call dscal\n";
}

//dnrm2 : vector 2-norm
extern "C" double dnrm2_(const int *N, 
			 const double *X, const int *INCX);

inline double dnrm2(const int& N, 
		    const double* X, const int& INCX){
  return dnrm2_(&N, X, &INCX);
  //std::cout << " - call dnrm2\n";
  //return 0.0;
} 
		  
//dcopy : copy one vector to another
extern "C" void dcopy_(const int *N, 
		       const double *DX, const int *INCX, 
		       double *DY, const int *INCY);

inline void dcopy(const int& N, 
		  const double* DX, const int& INCX,
		  double* DY, const int& INCY){
  dcopy_(&N, DX, &INCX, DY, &INCY);
  //std::cout << " - call dcopy\n";
}



//ddot : vector dot product
extern "C" double ddot_(const int *N, 
			const double *DX, const int *INCX,
			const double *DY, const int *INCY);

inline double ddot(const int& N, 
		   const double* DX, const int& INCX,
		   const double* DY, const int& INCY){
  return ddot_(&N,DX,&INCX,DY,&INCY);
  //std::cout << " - call ddot\n";
  //return 0.0;
}

//daxpy : vector addition
extern "C" void daxpy_(const int *N, const double *ALPHA, 
		       const double *X, const int *INCX, 
		       double *Y, const int *INCY);

inline void daxpy(const int& N, const double& ALPHA,
		  const double* X, const int& INCX,
		  double* Y, const int& INCY){
  daxpy_(&N,&ALPHA,X,&INCX,Y,&INCY);
  //std::cout << " - call daxpy\n";
}


/******** Level-2 routines ********/

//dgemv : matrix-vector product
extern "C" void dgemv_(const char *TRANS, const int *M, const int *N, 
		       const double *ALPHA, const double *A, const int *LDA, 
		       const double *X, const int *INCX,
		       const double *BETA, double *Y, const int *INCY);

inline void dgemv(bool TRANS, const int& M, const int& N,
		  const double& alpha, const double* A, const int& LDA,
		  const double* X, const int& INCX,
		  const double& BETA, double* Y, const int& INCY){
  char TCHAR = TRANS ? 'T' : 'N';
  dgemv_(&TCHAR, &M, &N, &alpha, A, &LDA, X, &INCX, &BETA, Y, &INCY);
  //std::cout << " - call dgemv\n";
}
		  
//dsymv : symmetric matrix-vector product
extern "C" void dsymv_(const char *UPLO, const int *N, const double *ALPHA,
		       const double *A, const int *LDA, const double *X,
		       const int *INCX, const double *BETA, double *Y,
		       const int *INCY);

inline void dsymv(bool Upper, const int& N, const double& ALPHA,
		  const double *A, const int& LDA, 
		  const double* X, const int& INCX,
		  const double& BETA,
		  double *Y, const int& INCY){
  char UCHAR = Upper ? 'U' : 'L';
  dsymv_(&UCHAR, &N, &ALPHA, A, &LDA, X, &INCX, &BETA, Y, &INCY);
  //std::cout << " - call dsymv\n";
}


/******** Level-3 routines ********/

//dgemm : matrix-matrix product
extern "C" void dgemm_(const char *TRANSA, const char *TRANSB, 
		       const int *M, const int *N, const int *K, 
		       const double *ALPHA, const double *A, 
		       const int *LDA, const double *B, const int *LDB, 
		       const double *BETA, double *C, const int *LDC);

inline void dgemm(bool TRANSA, bool TRANSB, 
		  const int& M, const int&N, const int& K,
		  const double& ALPHA, const double* A, const int& LDA,
		  const double* B, const int& LDB, const double& BETA,
		  double* C, const int& LDC){
  char ACHAR = TRANSA ? 'T' : 'N';
  char BCHAR = TRANSB ? 'T' : 'N';
  dgemm_(&ACHAR,&BCHAR,&M,&N,&K,&ALPHA,A,&LDA,B,&LDB,&BETA,C,&LDC);
  //std::cout << " - call dgemm\n";
}

//dsymm : symmetric matrix-matrix product
extern "C" void dsymm_(const char* SIDE,const char* UPLO,
		       const int* M, const int* N, const double* ALPHA,
		       const double* A, const int* LDA,
		       const double* B, const int* LDB,
		       const double* BETA, double* C, const int* LDC);

inline void dsymm(bool LEFT, bool UPPER, const int& M, const int& N, 
		  const double& ALPHA,
		  const double* A, const int& LDA,
		  const double* B, const int& LDB,
		  const double& BETA, double* C, 
		  const int& LDC){
  char LRCHAR = LEFT ? 'L' : 'R';
  char ULCHAR = UPPER ? 'U' : 'L';
  dsymm_(&LRCHAR,&ULCHAR,&M,&N,&ALPHA,A,&LDA,B,&LDB,&BETA,C,&LDC);
  //std::cout << " - call dsymm\n";
}


//dsyrk : symmetric rank-k update
extern "C" void dsyrk_(const char* UPLO, const char* TRANS,
		       const int* N, const int* K, const double* ALPHA, 
		       const double* A, const int* LDA, 
		       const double* BETA, double* C, const int* LDC);

inline void dsyrk(bool UPPER, bool TRANS, const int& N, const int& K, 
		  const double& ALPHA, const double* A, const int& LDA,
		  const double& BETA, double* C, const int& LDC){
  char TCHAR = TRANS ? 'T' : 'N';
  char UCHAR = UPPER ? 'U' : 'L';
  dsyrk_(&UCHAR,&TCHAR,&N,&K,&ALPHA,A,&LDA,&BETA,C,&LDC);
  //std::cout << " - call dsyrk\n";
}

//dgecv : matrix covariance (my own routine)
extern "C" void dgecv_(const char *INOUT, const int *M, const int *N, 
		       const double *A, const int *LDA,
		       double *C, const int *LDC);

inline void dgecv(const char& INOUT, const int& M, const int& N,
		  const double* A, const int& LDA,
		  double* C, const int& LDC){
  dgecv_(&INOUT,&M,&N,A,&LDA,C,&LDC);
  //std::cout << " - call dgecv\n";
}

		    

#endif /* BLAS_H */
