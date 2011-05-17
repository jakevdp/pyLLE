#ifndef LAPACK_H
#define LAPACK_H

#include <sstream>

/************************************************************
 LAPACK routines & low-level wrappers
************************************************************/

//dgetrf : LU factorization
extern "C" void dgetrf_(const int *M, const int *N, double *A, 
			const int *LDA, int *IPIV, int *INFO);

inline void dgetrf(int M, int N, double* A,
		   int LDA, int* IPIV){
  int INFO;
  dgetrf_(&N,&M,A,&LDA,IPIV,&INFO);
  if(INFO){
    std::cerr << INFO << "\n";
    throw MatVecError("Error in dgetrf");
  }
}

//dgetrs : solve a system of linear equations
//         using the result of dgetrf
//             A*X=B or A'*X=B
extern "C" void dgetrs_(char *TRANS, int *N, int *NRHS, double *A,
			int* LDA, int *IPIV, double *B, int *LDB, int *INFO);

inline void dgetrs(bool Trans, int N, int NRHS, double* A,
		   int LDA, int* IPIV, double* B, int LDB){
  int INFO;
  char T_CHAR = Trans ? 'T' : 'N';
  dgetrs_(&T_CHAR,&N,&NRHS,A,&LDA,IPIV,B,&LDB,&INFO);
  if(INFO){
    std::ostringstream oss;
    oss << "Error in dgetrs (INFO=" << INFO << ")\n";
    throw MatVecError(oss.str());
  }
}

//dsytrf : LU factorization, symmetric matrix
extern "C" void dsytrf_(const char* UPLO, const int *N, double *A, 
			const int *LDA, int *IPIV, double* WORK,
			const int* LWORK, int *INFO);

inline void dsytrf(bool upper, int N, double* A, int LDA, int*IPIV){
  int INFO;
  char UPLO = upper ? 'U' : 'L';
  
  //determine the optimal size of WORK array
  double WORK_QUERY;
  int LWORK = -1;
  dsytrf_(&UPLO,&N,A,&LDA,IPIV,&WORK_QUERY,&LWORK,&INFO);
  
  LWORK = int(WORK_QUERY);
  double* WORK = new double[LWORK];

  dsytrf_(&UPLO,&N,A,&LDA,IPIV,WORK,&LWORK,&INFO);
  if(INFO){
    std::cout << "DSYTRF error code = " << INFO << "\n";
    throw MatVecError("Error in dsytrf");
  }

  delete[] WORK;
}

//dsytrs : solve a system of linear equations
//         using the result of dsytrf
//             A*X=B or A'*X=B
//         with A symmetric
extern "C" void dsytrs_(char *UPLO, int *N, int *NRHS, double *A,
			int* LDA, int *IPIV, double *B, int *LDB, int *INFO);

inline void dsytrs(bool Upper, int N, int NRHS, double* A,
		   int LDA, int* IPIV, double* B, int LDB){
  int INFO;
  char UPLO = Upper ? 'U' : 'L';
  dsytrs_(&UPLO, &N, &NRHS, A, &LDA, IPIV, B, &LDB, &INFO);
  if(INFO){
    std::cerr << INFO << "\n";
    throw MatVecError("Error in dsytrs");
  }
}

//dgesdd : compute the singular value decomposition of a real matrix
//         optionally compute left/right singular vectors
extern "C" void dgesdd_(char *JOBZ, int *M, int *N, double *A, int *LDA, 
			double *S, double *U, int *LDU, double *VT, int *LDVT,
			double *WORK, int *LWORK, int *IWORK, int *INFO);

inline void dgesdd(char JOBZ, int M, int N,
		   double* A, int LDA, double* S, double* U,
		   int LDU, double* VT, int LDVT){
  int INFO;
  int* IWORK = new int[8 * std::min(M,N)];
  
  //determine optimal LWORK. This is done by setting LWORK = -1, 
  //  then WORK[0] gives the optimal LWORK
  int LWORK = -1;
  double WORK_TMP = 0;
  
  dgesdd_(&JOBZ, &M, &N, A, &LDA, S, U, &LDU, VT, 
	  &LDVT, &WORK_TMP, &LWORK, IWORK, &INFO);
  
  LWORK = int(WORK_TMP);
  double *WORK = new double[LWORK];
  
  //now compute the svd
  dgesdd_(&JOBZ, &M, &N, A, &LDA, S, U, &LDU, VT, 
	  &LDVT, WORK, &LWORK, IWORK, &INFO);
  
  if(INFO){
    std::cerr << INFO << "\n";
    throw MatVecError("Error in dgesdd");
  }

  delete[] WORK;
  delete[] IWORK;
}

//dgeqrf : compute a QR factorization of a matrix A
extern "C" void dgeqrf_(int *M, int *N, double *A, int *LDA, double *TAU, 
			double *WORK, int *LWORK, int *INFO);

inline void dgeqrf(int M, int N, double* A, int LDA,
		   double* TAU){
  int INFO;
  int LWORK = 3*N;
  double *WORK = new double[LWORK];
  dgeqrf_(&M,&N,A,&LDA,TAU,WORK,&LWORK,&INFO);
  if(INFO){
    std::cerr << INFO << "\n";
    throw MatVecError("Error in dgeqrf");
  }
  delete[] WORK;
}

//dorgqr_ : generates the real matrix Q for the QR decomposition
//           using the output of dgeqrf
extern "C" void dorgqr_(int *M, int *N, int *K, double *A, int *LDA,
			double *TAU, double *WORK, int *LWORK, int *INFO);

inline void dorgqr(int M, int N, int K, double* A,
		   int LDA,double* TAU){
  int INFO;
  int LWORK = 3*N;
  double *WORK = new double[LWORK];
  dorgqr_(&M,&N,&K,A,&LDA,TAU,WORK,&LWORK,&INFO);
  if(INFO){
    std::cerr << INFO << "\n";
    throw MatVecError("Error in dorgqr");
  }
  delete[] WORK;
}

//dsyev_ : compute eigenvalues and eigenvectors of a real symmetric matrix
extern "C" void dsyev_(char* JOBZ, char* UPLO, int* N, double* A,
		       int* LDA, double* W, double* WORK, int* LWORK,
		       int* INFO);

inline void dsyev(bool compute_v, bool upper, int N, double* A,
		  int LDA, double* W){
  int INFO;
  char JOBZ = compute_v ? 'V' : 'N';
  char UPLO = upper ? 'U' : 'L';
  
  //compute optimized LWORK
  int LWORK = -1;
  double WORK_TMP;
  dsyev_(&JOBZ,&UPLO,&N,A,&LDA,W,&WORK_TMP,&LWORK,&INFO);
  
  LWORK = int(WORK_TMP);
  double *WORK = new double[LWORK];
  
  dsyev_(&JOBZ,&UPLO,&N,A,&LDA,W,WORK,&LWORK,&INFO);
  
  delete[] WORK;
}



#endif /*LAPACK_H*/
