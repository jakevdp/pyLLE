#ifndef IRWPCA_H
#define IRWPCA_H

#include "MatVec.h"
#include "MatSym.h"
#include "MatVecDecomp.h"
#include <exception>
#include <string>
#include <sstream>


/******************************************************
PCA

 Inputs:
  - data is a [d_in x N] matrix
  - mu is a length [d_in] vector
  - trans is a [d_out x d_in] matrix
  - var (optional) gives the desired variance

 On Return:
  - mu is the centroid of data
  - trans and data_proj (optional) are such that
      data_proj = trans * (data - mu)
  - if (0<var<=1) then the variance is used to compute
     desired d_out.
******************************************************/
void PCA(const Matrix<double>& data,
	 Vector<double>& mu,
	 Matrix<double>& trans,
	 double var=0.0);

void PCA(const Matrix<double>& data,
	 Vector<double>& mu,
	 Matrix<double>& trans,
	 Matrix<double>& data_proj,
	 double var=0.0);

/******************************************************
WPCA

 Inputs:
  - data is a [d_in x N] matrix
  - mu is a length [d_in] vector
  - trans is a [d_out x d_in] matrix
  - A is a length [N] matrix of weights 
       for each point in data
  - var (optional) gives the desired variance

 On Return:
  - mu is the weighted centroid of data
  - trans and data_proj (optional) are such that
      data_proj = trans * (data - mu)
  - if (0<var<=1) then the variance is used to compute
     desired d_out.
******************************************************/
void WPCA(const Matrix<double>& data,
	  const Vector<double>& A,
	  Vector<double>& mu,
	  Matrix<double>& trans,
	  double var=0.0);

void WPCA(const Matrix<double>& data,
	  const Vector<double>& A,
	  Vector<double>& mu,
	  Matrix<double>& trans,
	  Matrix<double>& data_proj,
	  double var=0.0);



/******************************************************
IRWPCA - Iteratively Reweighted PCA
-------------------------------------------------------
 Inputs:
  - data is a [d_in x N] matrix
  - mu is a length [d_in] vector
  - trans is a [d_out x d_in] matrix
  - A is a length [N] matrix of weights 
       for each point in data
  - tol is a number such that x<tol is sufficiently close to zero
  - MAX_ITER is the maximum number of iterations allowed.
      if convergence is not reached within this number of
      iterations, execution will terminate

 On Return:
  - A are the weights
  - mu is the weighted centroid of data
  - trans is the transform such that
      data_proj = trans * (data - mu)

    ----------------------------------

 based on algorithm outlined in
     'Robust Locally Linear Embedding'
      by Hong Chang & Dit-Yan Yeung, 2005

******************************************************/
void IRWPCA(const Matrix<double>& data,
	    Vector<double>& weights,
	    Vector<double>& mu,
	    Matrix<double>& trans,
	    const double tol=1E-8,
	    const int MAX_ITER=1000,
	    const bool quiet=true);



/***********************************************************
  make_weights:
    function used by IRWPCA.  
     On input, A is a vector of squared errors from PCA reconstruction.
     On output, it is the associated weights for WPCA
************************************************************/
void make_weights(Vector<double>& A);


/***********************************************************
  IterException:
    a custom exception class for when IRWPCA reaches
    maximum iterations
************************************************************/
class IterException: public std::exception
{
public:
  IterException(const std::string& s = "IRWPCA: Max iterations reached."){msg=s;}
  IterException(const int MAXITER, const double tol=0.0)
    {
      std::stringstream s;
      s << "IRWPCA: Max iterations (" << MAXITER << ") reached";
      if(tol>0)
	s<<" before reaching tol = " << tol;
      s << std::endl;
      msg = s.str();
    }
  ~IterException() throw() {}
  virtual const char* what() const throw(){return msg.c_str();}
private:
  std::string msg;
};

#endif // IRWPCA_H
