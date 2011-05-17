#ifndef MATVECDECOMP_H
#define MATVECDECOMP_H

#include "MatVec.h"
#include "MatSym.h"

/************************************************************
 * SVD class to perform singular value decomposition
 *  of a matrix using the LAPACK interface 
 ************************************************************/
class SVD{
 public:
  SVD(Matrix<double,DENSE>& A, 
      bool compute_uv = true,
      bool overwrite = false,
      bool full_matrices = false);
  Matrix<double,DENSE> U;
  Vector<double,GEN> S;
  Matrix<double,DENSE> VT;
};

/************************************************************
 * LUD class to perform LU-decomposition of a matrix 
 *  using the LAPACK interface 
 ************************************************************/
class LUD{
 public:
  LUD(const Matrix<double,DENSE>& A);
  LUD(const Matrix<double,SYM>& A);
  Matrix<double,DENSE> LU;
  Vector<int> IPIV;
  bool flag() const{return flag_;}
  bool sym() const{return sym_;}
 private:
  bool flag_; //if symmmetric, flag = upper
              //otherwise flag = transpose
  bool sym_;
};

/************************************************************
 * QRD class to perform QR-decomposition of a matrix 
 *  using the LAPACK interface 
 ************************************************************/
class QRD{
 public:
  QRD(Matrix<double,DENSE>& A);
  Matrix<double,DENSE> Q;
  Matrix<double,DENSE> R;
};



/************************************************************
 * EIGS class to find specified eigenvalues/vectors
 *  of a symmetric matrix using the LAPACK interface 
 ************************************************************/
class EIGS{
public:
  EIGS(Matrix<double,SYM>& A, bool compute_v = true);
  Vector<double,GEN> evals;
  Matrix<double,DENSE> evecs;
};





/************************************************************
 * EIGS_AR class to find specified eigenvalues/vectors
 *  of a symmetric matrix using the ARPACK interface 
 ************************************************************/
class EIGS_AR{
public:
  EIGS_AR(Matrix<double,SYM>& A, 
       int nev, const std::string& which="SM",
       int num_to_ignore=0, bool quiet=false);
  
  EIGS_AR(Matrix<double,SYM>& A, const Matrix<double,SYM>& B, 
       int nev, const std::string& which="SM",
       int num_to_ignore=0,bool quiet=false);
  
  Matrix<double,DENSE> evecs;
  Vector<double,GEN> evals;
private:
  void compute_eigs(Matrix<double,SYM>& A, const Matrix<double,SYM>& B, 
		    const std::string& which,
		    int num_to_ignore, bool general);
  bool quiet_;
};


/************************************************************
  SOLVE:

  wrapper for the LAPACK DGESDD/DGETRS routine 
   to solve a linear system of equations

  A(^T) * X = B   (X,B are matrices)

    or

  A(^T) * x = b   (x,b are vectors)

  BX is B on input, solution (X) on output

  A can be either a matrix, or an LUD object.
    if A is a matrix, the LUD will be calculated.
************************************************************/


template<class MType>
void SOLVE(MType& A, 
	    Matrix<double,DENSE>& BX){
  if(A.nrows()!=A.ncols())
    throw MatVecError("SOLVE works only for sqare A\n");
  LUD A_LU(A);
  SOLVE(A_LU,BX);
}

template<class MType>
void SOLVE(MType& A, 
	   Matrix<double,DENSE>& X,
	   const Matrix<double>& B){
  if(A.nrows()!=A.ncols())
    throw MatVecError("SOLVE works only for sqare A\n");
  LUD A_LU(A);
  SOLVE(A_LU,X,B);
}

template<class MType>
void SOLVE(MType& A, 
	    Vector<double,GEN>& bx){
  if(A.nrows()!=A.ncols())
    throw MatVecError("SOLVE works only for sqare A\n");
  LUD A_LU(A);
  SOLVE(A_LU,bx);
}

template<class MType>
void SOLVE(MType& A, 
	    Vector<double,GEN>& x,
	    const Vector<double,GEN>& b){
  if(A.nrows()!=A.ncols())
    throw MatVecError("SOLVE works only for sqare A\n");
  LUD A_LU(A);
  SOLVE(A_LU,x,b);
}

template<>
void SOLVE(LUD& A, 
	   Matrix<double,DENSE>& BX);
template<>
void SOLVE(LUD& A, 
	   Matrix<double,DENSE>& X,
	   const Matrix<double,DENSE>& B);
template<>
void SOLVE(LUD& A, 
	   Vector<double,GEN>& bx);
template<>
void SOLVE(LUD& A, 
	    Vector<double,GEN>& x,
	    const Vector<double,GEN>& b);

#endif /* MATVECDECOMP_H */
