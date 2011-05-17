#include "IRWPCA.h"
#include <iostream>
#include <math.h>



void PCA(const Matrix<double>& data,
	 Vector<double>& mu,
	 Matrix<double>& trans,
	 Matrix<double>& data_proj,
	 double var/*=0.0*/)
{
  Vector<double> A(data.ncols(),1.0);
  WPCA(data,A,mu,trans,data_proj,var);
}

void PCA(const Matrix<double>& data,
	 Vector<double>& mu,
	 Matrix<double>& trans,
	 double var/*=0.0*/)
{
  Vector<double> A(data.ncols(),1.0);
  WPCA(data,A,mu,trans,var);
}


void WPCA(const Matrix<double>& data,
	  const Vector<double>& A,
	  Vector<double>& mu,
	  Matrix<double>& trans,
	  double var/*=0.0*/)
{
  int d_out = 0;
  bool use_variance = (var>0);
  int d_in = data.nrows();
  int N = data.ncols();

  if(mu.size() != d_in)
    mu.reallocate(d_in);

  if(use_variance and var>1)
    throw MatVecError("WPCA : var must be between 0 and 1\n");
  
  if(!use_variance){
    d_out = trans.nrows();
  }
  
  //compute weighted mean
  for(int i=0;i<d_in;i++)
    {
      mu(i)=0.0;
      for(int j=0;j<N;j++) 
	mu(i) += A(j) * data(i,j);
    }
  mu /= A.SumElements();
  
  //M is a centered and weighted version of data
  Matrix<double> M(d_in,N);
  for(int i=0;i<N;i++){
    M.col(i) = sqrt(A(i))*(data.col(i)-mu);
  }
  
  //Find trans with an SVD
  SVD M_SVD(M,true,true); //overwrite M

  if(use_variance){
    //compute d_out from SVD
    double total=0;
    double Sum = M_SVD.S.SumElements();
    d_out = 0;
    while(total<Sum){
      total+=M_SVD.S(d_out)/Sum;
      d_out++;
    }
  }

  if(trans.ncols() != d_in || trans.nrows() != d_out)
    trans.reallocate(d_out,d_in);
    
  for(size_t i=0;i<d_out;i++){
    trans.row(i) = M_SVD.U.col(i);
  }
}


void WPCA(const Matrix<double>& data,
	  const Vector<double>& A,
	  Vector<double>& mu,
	  Matrix<double>& trans,
	  Matrix<double>& data_proj,
	  double var/*=0.0*/)
{
  int d_out;
  bool use_variance = (var>0);
  int d_in = data.nrows();
  int N = data.ncols();

  if(use_variance and var>1)
    throw MatVecError("WPCA : var must be between 0 and 1\n");
  
  if(!use_variance){
    d_out = trans.nrows();
  }

  if(mu.size() != d_in)
    mu.reallocate(d_in);
  
  //compute weighted mean
  for(int i=0;i<d_in;i++)
    {
      mu(i)=0.0;
      for(int j=0;j<N;j++) 
	mu(i) += A(j) * data(i,j);
    }
  mu /= A.SumElements();
  
  //M is a centered and weighted version of data
  Matrix<double> M(d_in,N);
  for(int i=0;i<N;i++){
    M.col(i) = sqrt(A(i))*(data.col(i)-mu);
  }
  
  //Find trans with an SVD
  SVD M_SVD(M,true,true); //overwrite M

  if(use_variance){
    //compute d_out from SVD
    double total=0;
    double Sum = M_SVD.S.SumElements();
    d_out = 0;
    while(total<var){
      total+=M_SVD.S(d_out)/Sum;
      d_out++;
    }
  }

  if(trans.ncols() != d_in || trans.nrows() != d_out)
    trans.reallocate(d_out,d_in);

  if(data_proj.nrows() != d_out || data_proj.ncols() != N)
    data_proj.reallocate(d_out,N);
    
  //trans = M_SVD.U.Cols(0,d_out).Transpose();
  for(size_t i=0;i<d_out;i++){
    trans.row(i) = M_SVD.U.col(i);
  }

  //data_proj = M_SVD.S * M_SVD.VT;
  for(size_t i=0;i<d_out;i++){
    data_proj.row(i) = M_SVD.S(i) * M_SVD.VT.row(i);
  }
}


void IRWPCA(const Matrix<double>& data,
	    Vector<double>& weights,
	    Vector<double>& mu,
	    Matrix<double>& trans,
	    const double tol/*=1E-8*/,
	    const int max_iter/*=1000*/,
	    const bool quiet/* = true*/)
{
  int d_in = data.nrows();
  int d_out = trans.nrows();
  int N = data.ncols();
  
  //check on input data
  if (mu.size() != d_in)
    mu.reallocate(d_in);

  if (trans.ncols() != d_in)
    trans.reallocate(d_out,d_in);
  
  if (weights.size() != N)
    weights.reallocate(N,1.0);
  else
    weights.SetAllTo(1.0);

  if (max_iter<1)
    throw MatVecError("max_iter must be positive");
  
  //data needed for procedure
  Matrix<double,SYM> transcov(d_in,d_in);
  Vector<double> V;
  Vector<double> weights_old;
  Vector<double> V_minus_mu;
  Matrix<double> trans_old;

  PCA(data,mu,trans);

  //Now we iterate and find the optimal weights,
  // mean, and transformation
  for(int count=1;count<max_iter;count++)
    {
      weights_old.deep_copy(weights);
      trans_old.deep_copy(trans);

      transcov = trans.Transpose() * trans;
      
      //find errors and new weights
      for(int i=0;i<N;i++)
	{
	  V.deep_copy( data.col(i) );
	  V -= mu;
	  V_minus_mu.deep_copy(V);
	  V -= transcov * V_minus_mu;
	  weights(i) = pow( blas_NRM2(V), 2 );
	}     
      
      make_weights(weights);
      WPCA(data,weights,mu,trans);
      
      //check for convergence using residuals between
      //  new weights and old weights
      weights_old -= weights;
      double Wres = blas_NRM2(weights_old);
      //  and new trans and old trans
      //trans_old -= trans;
      //double Tres = blas_NRM2(trans_old);
	
      if ( ( Wres/sqrt(N) <tol) )
	{
	  if (!quiet)
	    std::cout << "IRWPCA: converged to tol = " << tol 
		      << " in " << count << " iterations\n";
	  return;
	}
    }//end for

  //if tolerance was not reached, return an error message
  throw IterException(max_iter,tol);
}

void make_weights(Vector<double>& A)
  //on input, A is a vector of squared errors
  //on output, it is the associated weights
{
  int N = A.size();
  double c = 0.5 * A.SumElements() / N;
  double S = 0.0;
  for(int i=0;i<N;i++)
    {
      if(A(i)>c) 
	A(i) = c/A(i);
      else 
	A(i) = 1.0;
      S += A(i);
    }
  A /= S;
}
