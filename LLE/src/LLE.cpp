#include "argsort.h"
#include "LLE.h"
#include "LLE_Except.h"
#include "MatVec.h"
#include "MatSym.h"
#include "MatVecDecomp.h"
#include "IRWPCA.h"
//#include "BallTree.h"
#include <math.h>
#include <stdlib.h>
#include <time.h>

void compute_neighbors(const Mat_data_t& training_data, 
		       Mat_index_t& neighbors,
		       bool use_tree,
		       int verbose){
  compute_neighbors(training_data,
		    training_data,
 		    neighbors,use_tree,verbose);
}


void compute_neighbors(const Mat_data_t& training_data, 
		       const Mat_data_t& test_data,
		       Mat_index_t& neighbors, 
		       bool use_tree,
		       int verbose){
  if(test_data.ncols() != neighbors.ncols()){
    throw LLE_Error("compute_neighbors : test_data and nbrs "
		      "must have same ncols");
  }
  if(test_data.nrows() != training_data.nrows()){
    throw LLE_Error("compute_neighbors : test_data and training_data "
		      "must have same leading dimension");
  }
  if(neighbors.nrows() > training_data.ncols()){
    throw LLE_Error("compute_neighbors : cannot have more neighbors than training points");
  }

  if(use_tree){
    std::cerr << "LLE_neighbors: BallTree not implemented; using brute force\n";
  }
  clock_t start = 0;
  clock_t end = 0;
  
  /*
  int k = neighbors.nrows();
  int Ntrain = training_data.ncols();
  int Ntest = test_data.ncols();
  int D = training_data.nrows();

  //----------------------------------------
  // Find neighbors with Ball Tree
  if(use_tree){
    if(verbose > 0){
      std::cout << "Constructing Ball Tree for " << Ntrain
		<< " points in " << D << " dimensions:\n";
      start = clock();
    }
    BallTree<double> BT(training_data);
    if(verbose > 0){
      end = clock();
      std::cout << "   BallTree construction: Finished in "  
		<< (end-start)*1.0/CLOCKS_PER_SEC << " sec\n";
      std::cout << "Finding Neighbors:\n";
      start = clock();
    }
    if(&training_data == &test_data){
      //we must ignore the first neighbor
      for(int i=0;i<Ntest;i++){
	Vec_index_t nbrs_i(k+1);
	BT.knn_search(training_data.col(i),nbrs_i);
	neighbors.col(i) = nbrs_i.SubVector(1,k+1);
	if( nbrs_i(0) != i){
	  for(int j=0;j<k;j++)
	    if(neighbors(j,i)==i){
	      neighbors(j,i)= nbrs_i(0);
	      break;
	    }
	}
      }
    }else{
      for(int i=0;i<Ntest;i++){
	Vec_index_t nbrs_i = neighbors.col(i);
	BT.knn_search(test_data.col(i),nbrs_i);
      }
    }
    if(verbose>0){
      end = clock();
      std::cout << "   BallTree neighbors: Finished in " 
		<< (end-start)*1.0/CLOCKS_PER_SEC << " sec\n";
    }
  }
  
  
  //----------------------------------------
  // Find neighbors with Brute Force

  else{
  */
  //if test data and training data are the same, then the nearest
  // neighbor of a point is itself.  Use a variable starting_k to
  // correct for this
  
  int starting_k = ( (&test_data) == (&training_data) ) ? 1 : 0;
  
  int d_in = training_data.nrows();
  int k = neighbors.nrows();
  size_t Ntest = test_data.ncols();
  size_t Ntrain = training_data.ncols();

  if(verbose>0)
    std::cout << "compute_neighbors() : computing " << k << " neighbors "
	      << " of "<< Ntest << " points in " << d_in << " dimensions\n";
  
  Mat_data_t data_i(d_in,Ntrain);
  Vec_data_t D(Ntrain);
  Vec_index_t indices(Ntrain);
  
  //temporary vectors for arithemetic
  Vec_data_t colj;
  Vec_data_t test_coli;
  Vec_data_t train_colj;
  Vec_index_t ncoli;
  Vec_index_t iSV;
  
  if(verbose>0)
    start = clock();
  
  for(size_t i=0;i<Ntest;i++)
    {
      //find k nearest neighbors of test_data[i] within training_data
      for (size_t j=0;j<Ntrain;j++){
	data_i.col(j) = test_data.col(i) - training_data.col(j);
	D(j) = blas_NRM2( data_i.col(j) );
      }
      // find k nearest points
      argsort(D,indices,k+starting_k);
      neighbors.col(i) = indices.SubVector(starting_k,starting_k+k);
    }
  if(verbose>0){
    end = clock();
    std::cout << "   brute-force neighbors: Finished in " 
	      << (end-start)*1.0/CLOCKS_PER_SEC << " sec\n";
  }
}  

void compute_LLE_weights(const Mat_data_t& training_data,
			 int k,
			 Mat_data_t& weight_matrix,
			 int verbose){
  Mat_index_t neighbors( k,training_data.ncols() );
  compute_neighbors(training_data,neighbors,false,verbose);
  compute_LLE_weights(training_data,training_data,neighbors,
		      weight_matrix,verbose);
}


void compute_LLE_weights(const Mat_data_t& training_data,
			 const Mat_index_t& neighbors,
			 Mat_data_t& weight_matrix,
			 int verbose){
  compute_LLE_weights(training_data,training_data,neighbors,
		      weight_matrix,verbose);
}


void compute_LLE_weights(const Mat_data_t& training_data,
			 const Mat_data_t& test_data,
			 int k,
			 Mat_data_t& weight_matrix,
			 int verbose){
  Mat_index_t neighbors( k,test_data.ncols() );
  compute_neighbors(training_data,test_data,neighbors,false,verbose);
  compute_LLE_weights(training_data,test_data,neighbors,weight_matrix,verbose);
}


void compute_LLE_weights(const Mat_data_t& training_data,
			 const Mat_data_t& test_data,
			 const Mat_index_t& neighbors,
			 Mat_data_t& weight_matrix,
			 int verbose){
  unsigned int Ntrain = training_data.ncols();
  unsigned int Ntest = test_data.ncols();
  if(test_data.ncols() != neighbors.ncols()){
    throw LLE_Error("compute_LLE_weights : test_data and nbrs "
		      "must have same ncols");
  }
  if(test_data.nrows() != training_data.nrows()){
    throw LLE_Error("compute_LLE_weights : test_data and training_data "
		      "must have same leading dimension");
  }
  if(weight_matrix.nrows() != Ntest || weight_matrix.ncols() != Ntrain){
    throw LLE_Error("compute_LLE_weights : weight matrix has wrong dimensions");
  }
  int d_in = test_data.nrows();
  int k = neighbors.nrows();
  
  if(verbose > 0)
    std::cout << "compute_LLE_weights()\n";
  if(verbose > 1)
    std::cout << " - Constructing [" 
	      << Ntest << " x " << Ntrain 
	      << "] weight matrix.\n";

  weight_matrix.SetAllTo(0.0);

  Mat_data_t neighborhood(d_in,k);
  Matrix<double,SYM> Q(k);
  Vec_data_t w(k);
  
  for (size_t i=0;i<Ntest;i++)
    {
      for(int j=0;j<k;j++){
	neighborhood.col(j) =
	  training_data.col( neighbors(j,i) ) - test_data.col(i);
      }
      
      //Construct the [k x k] covariance matrix 
      //   of the neighborhood
      Q = neighborhood.Transpose() * neighborhood;
      
      //add a fraction of the trace to the diagonal
      // this prevents matrix from being singular
      //Q += 0.001 * float(Q.Trace());
      Q.diag() += 0.001*Q.Trace();
      
      //solve for w in Q*w = [1,1,1...1]^T
      //  and put into weight matrix
      w.SetAllTo(1.0);

      SOLVE(Q,w);
      w /= w.SumElements();
      
      for(int j=0;j<k;j++)
	weight_matrix(i,neighbors(j,i)) = w(j);
    }
}

int compute_LLE_dim(const Mat_data_t& training_data,
		    int k,
		    double var,
		    int verbose){
  Mat_index_t neighbors( k,training_data.ncols() );
  compute_neighbors(training_data, neighbors, false, verbose);
  return compute_LLE_dim(training_data,neighbors,var,verbose);  
}

int compute_LLE_dim(const Mat_data_t& training_data,
		    const Mat_index_t& neighbors,
		    double var,
		    int verbose){
  if( training_data.ncols() != neighbors.ncols() )
    throw LLE_Error("compute_LLE_dim : neighbors and training_data must have same number of points");
  if( var<=0 ||var > 1)
    throw LLE_Error("compute_LLE_dim : variance must satisfy 0 < var <= 1");

  double var_total = 0.0;
  size_t d_in = training_data.nrows();
  size_t N = training_data.ncols();
  size_t k = neighbors.nrows();
    
  if(verbose > 0)
    std::cout << "compute_LLE_dim():\n";
  if(verbose > 1)
    std::cout << " - Computing dimensionality of " << N <<" points in " 
	      << d_in << " dimensions\n"
	      << "    using " << k << " neighbors, with var = " 
	      << var << "\n";
  
  //------------------------------------------------
  // Find dimensionality at each neighborhood
  Mat_data_t neighborhood(d_in,k);
  Vec_index_t dim_array(N);
  
  Vec_data_t colj;
  Vec_data_t col_nji;
  Vec_data_t coli;
  
  for (size_t i=0;i<N;i++)
    {
      for(size_t j=0;j<k;j++){
	neighborhood.col(j) = 
	  training_data.col(neighbors(j,i) ) - training_data.col(i);
      }
      
      //find singular values of neighborhood
      // determine how many are needed to sum to var
      SVD nbrs_SVD(neighborhood,false);
      
      for(size_t j=0;j<nbrs_SVD.S.size();j++)
	nbrs_SVD.S(j) *= nbrs_SVD.S(j);
      
      nbrs_SVD.S /= nbrs_SVD.S.SumElements();
      
      //find number of dimensions of nbrs_SVD.S needed to add to var
      dim_array(i)=d_in;
      size_t dim;
      for (dim=1;dim<d_in;dim++){
	if (nbrs_SVD.S(dim-1)>var)
	  { 
	    dim_array(i)=dim;
	    break;
	  }
	nbrs_SVD.S(dim) += nbrs_SVD.S(dim-1);
      }
      var_total += nbrs_SVD.S(dim-1);
    }
  
  var_total /= N;
  
  //find mean and standard deviation of dim_array
  double d = 0.0;
  double d2 = 0.0;
  for(size_t i=0;i<N;i++){
    d += dim_array(i);
    d2 += pow( dim_array(i),2 );
  }
  d /= N;
  d2 /= N;
  float sig = sqrt(d2-d*d);
  
  //find d_out.  Round down if it will stay within sig of d
  //  round up otherwise
  int d_out;
  if ( (d<floor(d)+0.5) && (int(floor(d)) == int(ceil(d-sig))) )
    d_out = int(floor(d));
  else
    d_out = int(ceil(d));
  
  if(verbose > 1){
    std::cout << " - Intrinsic dimensionality = "
	      << d_out << " ("
	      << d << " +/- " 
	      << sig << ")\n";
    std::cout << "    for a variance of "
	      << var << " (avg " << var_total << " conserved)\n";
  }
  
  return d_out;
}

void LLE(const Mat_data_t& training_data,
	 int k,
	 Mat_data_t& projection,
	 int verbose){
  Mat_index_t neighbors(k,training_data.ncols());
  compute_neighbors(training_data,neighbors, false, verbose);
  LLE(training_data,neighbors,projection,verbose);
}

void LLE(const Mat_data_t& training_data,
	 const Mat_index_t& neighbors,
	 Mat_data_t& projection,
	 int verbose){
  size_t N = training_data.ncols();
  size_t d_in = training_data.nrows();
  size_t d_out = projection.nrows();

  if(projection.ncols()!= N)
    throw LLE_Error("LLE : projection must have same number of points as training_data");

  if(neighbors.ncols() != N)
    throw LLE_Error("LLE : neighbors must have same number of points as training_data");

  Mat_data_t W_I(N,N);
  compute_LLE_weights(training_data,neighbors,W_I,verbose);

  W_I.diag() -= 1.0;
  
  if(verbose > 0)
    std::cout << "LLE() :\n";
  if(verbose > 1)
    std::cout << " - Calculating LLE projection of " << N <<" points in "
	      << d_in << " -> " 
	      << d_out << " dimensions.\n";

  //------------------------------------------------
  //projection is the null space of (W-I)
  //  by the Rayleigh-Ritz theorem, this is given by the
  //   eigenvectors corresponding to the d+1 smallest
  //   eigenvalues of (W-I)^T * (W-I)
  if(verbose > 2)
    std::cout << "    + Constructing [" << N << " x " << N 
	      << "] Covariance Matrix.\n";
  Matrix<double,SYM> Cov = W_I.Transpose() * W_I;

  //now find eigenvectors of W...

  bool use_LAPACK = false;
  
  if(use_LAPACK){
    if(verbose > 1)
      std::cout << " - Finding null space of weight matrix with ARPACK\n";
    
    EIGS C_EIGS(Cov);
    
    if(verbose > 1) 
      std::cout << " - copying training proj\n";
    
    for(int i=0;i<projection.nrows();i++)
      projection.row(i) = C_EIGS.evecs.col(i+1);
  }else{
    if(verbose > 1)
      std::cout << " - Finding null space of weight matrix with ARPACK\n";
    
    if(verbose > 2)
      std::cout << "   + ";

    EIGS_AR C_EIGS(Cov,d_out,"SM",1,!(verbose>2) );
    
    if(verbose > 1) 
      std::cout << " - copying training proj\n";
    
    projection = C_EIGS.evecs.Transpose();
  }
  
  if(verbose > 1)
    std::cout << " - Success!!\n";
}

void HLLE(const Mat_data_t& training_data,
	  int k,
	  Mat_data_t& projection,
	  int verbose){
  Mat_index_t neighbors(k,training_data.ncols());
  compute_neighbors(training_data,neighbors, false, verbose);
  HLLE(training_data,neighbors,projection,verbose);
}

void HLLE(const Mat_data_t& training_data,
	  const Mat_index_t& neighbors,
	  Mat_data_t& projection,
	  int verbose){
  size_t N = training_data.ncols();
  size_t d_in = training_data.nrows();
  size_t d_out = projection.nrows();
  size_t k = neighbors.nrows();
  
  if(projection.ncols()!= N)
    throw LLE_Error("HLLE : projection must have same number of points as training_data");

  if(neighbors.ncols() != N)
    throw LLE_Error("HLLE : neighbors must have same number of points as training_data");

  if(k <= d_in)
    throw LLE_Error("HLLE : k must be greater than d_in");
  
  //----------------------------------------------------------------------
  //compute weight matrix
  int dp = ( d_out*(d_out+1) )/2;
  
  if (verbose>0)
    std::cout << "HLLE()\n"
	      << " - Performing HLLE on " << N <<" points in " 
	      << d_in << "->"<< d_out << " dimensions.\n";
  
  if(verbose > 2) 
    std::cout << "   + Constructing [" 
	      << dp*N << " x " << N
	      << "] weight matrix.\n";

  Mat_data_t W(dp*N,N);

  Mat_data_t neighborhood(d_in,k);
  Vec_data_t n_center(d_in);
  Mat_data_t Yi(k,1+d_out+dp);
  
  //D_mm: a diagonal matrix, with diagonal elements accessed
  // by D_mm_diag
  Mat_data_t D_mm(k,k,0.0);
  Vec_data_t D_mm_diag = D_mm.diag();
      
  Vec_data_t temp1;
  Vec_data_t temp2;
  Vec_data_t temp3;
  
  for (size_t i=0;i<N;i++)
    {
      //obtain all points in the neighborhood
      // and subtract their centroid
      n_center.SetAllTo(0.0);

      for(size_t j=0;j<k;j++)
	{
	  neighborhood.col(j) = training_data.col(neighbors(j,i));
	  n_center += neighborhood.col(j);
	}
      n_center /= k;
      
      for(size_t j=0;j<k;j++){
	neighborhood.col(j) -= n_center;
      }
      
      //Compute local coordinates using SVD
      SVD n_SVD(neighborhood);

      /*note: given NC = neighborhood.ncols()
       *             NR = neighborhood.nrows()
       * if NR >= NC, U is overwritten on neighborhood
       * otherwise, U is written to VT
      */
      
      /* Now construct Yi such that:
	  column 0 is all 1
	  column 1...d_out is V, where SVD of neighborhood = U * s * V^T
	  column d_out+1...d_out+dp is the hessian estimator 
	                            of the neighborhood */
      Yi.col(0).SetAllTo(1.0);

      for(size_t j=0; j<d_out; j++){
	Yi.col(j+1) = n_SVD.VT.row(j);
      }
      
      int count = 0;
      
      for(size_t mm=0;mm<d_out;mm++)
	for(size_t nn=mm;nn<d_out;nn++){
	  D_mm.diag() = Yi.col(mm+1);
	  Yi.col(1+d_out+count) = D_mm * Yi.col(nn+1);
	  count++;
	}
      
      /* Orthogonalize the linear and quadratic forms with
	 a QR factorization of Yi */
      QRD Yi_QRD(Yi);
      
      /* present w is given by the cols of Yi
	 between d_out+1 and dp+d_out+1 */
      
      /* Make cols of w sum to one */
      double S;
      for(size_t j=d_out+1; j<dp+d_out+1;j++){
	S =  temp1.SumElements();
	if(S<1E-4) break;
	
	Yi_QRD.Q.col(j) /= S;
      }
      
      /* Put weights in Weight matrix */
      for(size_t j=0;j<k;j++){
	W.col( neighbors(j,i) ).SubVector(i*dp,(i+1)*dp) =
	  Yi_QRD.Q.row(j).SubVector(d_out+1,dp+d_out+1);
      }
    }
  
  //-----------------------------------------------
  /* now we find the null space of W by finding the first
     d+1 eigenvectors of W^T * W */
  if(verbose > 2) 
    std::cout << "   + Constructing ["<<N<<" x "<<N
	      <<"] Covariance Matrix.\n";
  
  Matrix<double,SYM> C = W.Transpose() * W;
  
  //clear the memory
  W.reallocate(0,0);
  
  if (verbose > 1){
    std::cout << " - Finding null space of weight matrix with ARPACK\n";
  }

  if (verbose > 2)
    std::cout << "   + ";
  EIGS_AR C_EIGS(C,d_out,"SM",1,!(verbose > 2));

  //copy values to training_proj_
  projection = C_EIGS.evecs.Transpose();
  
  if (verbose > 1)
    std::cout << " - Normalizing null space with SVD\n";
  
  projection *= sqrt(N);
  
  /* now we need to normalize Y = projection
   * we need R = (Y.T*Y)^(-1/2)
   *   do this with an SVD of Y
   *      Y = U*sig*V.T
   *      Y.T*Y = (V*sig.T*U.T) * (U*sig*V.T)
   *            = V*(sig^2)*V.T
   *   so
   *      R = V * sig^-1 * V.T
   * 
   *   and our return value is
   *      Y*R
   */
  
  SVD proj_SVD(projection);
  
  Mat_data_t Smat(d_out,d_out,0.0);
  for(size_t i=0;i<d_out;i++)
    Smat(i,i) = 1.0 / proj_SVD.S(i);
  
  Mat_data_t R = projection * proj_SVD.VT.Transpose();
  Mat_data_t R2 = R*Smat;
  projection = R2 * proj_SVD.VT;
}

void MLLE(const Mat_data_t& training_data,
	  int k,
	  Mat_data_t& projection,
	  double TOL,
	  int verbose){
  Mat_index_t neighbors(k,training_data.ncols());
  compute_neighbors(training_data,neighbors, false, verbose);
  MLLE(training_data,neighbors,projection,TOL,verbose);
}

void MLLE(const Mat_data_t& training_data,
	  const Mat_index_t& neighbors,
	  Mat_data_t& projection,
	  double TOL,
	  int verbose){
  size_t N = training_data.ncols();
  size_t d_in = training_data.nrows();
  size_t d_out = projection.nrows();
  size_t k = neighbors.nrows(); 
  
  if(projection.ncols()!= N)
    throw LLE_Error("MLLE : projection must have same number of points as training_data");

  if(neighbors.ncols() != N)
    throw LLE_Error("MLLE : neighbors must have same number of points as training_data");
  
  if(verbose > 0)
    std::cout << "MLLE() :\n";
  if(verbose > 1)
    std::cout << " - Calculating MLLE projection of " << N <<" points in "
	      << d_in << " -> " 
	      << d_out << " dimensions.\n";
  
  //define some variables to hold values
  Vec_data_t rho(N);
  Mat_data_t w_reg(k,N,1.0);
  Mat_data_t evals(N,k,0.0);
  std::vector< Mat_data_t* > V(N);
  Matrix<double,SYM> Q(k);
  Vec_data_t w(k);
  Mat_data_t Gi(d_in,k);

  if(verbose > 2)
    std::cout << "   + computing LLE weights for all points\n";

  for(size_t i=0; i<N; i++){
    //find regularized weights: this is the same process as normal LLE

    for(size_t j=0;j<k;j++){
      Gi.col(j) =
	training_data.col( neighbors(j,i) ) - training_data.col(i);
    }
    
    //Construct the [k x k] covariance matrix 
    //   of the neighborhood
    Q = Gi.Transpose() * Gi;
      
    //add a fraction of the trace to the diagonal
    // this prevents matrix from being singular
    Q.diag() += 0.001*Q.Trace();
      
    //solve for w in Q*w = [1,1,1...1]^T
    //  and put into weight matrix
    Vec_data_t w = w_reg.col(i);
    w.SetAllTo(1.0);
    SOLVE(Q,w);

    w /= w.SumElements();

    //find the eigenvectors and eigenvalues of Gi.T*Gi
    // using SVD
    // we want V[i] to be a [k x k] matrix, where the columns
    // are eigenvectors of Gi^T * G
    
    //compute SVD, no overwrite, full matrices
    Mat_data_t GiT = Gi.Transpose();
    SVD G_SVD(GiT,true,false,true);
    
    V[i] = new Mat_data_t(0,0);
    V[i]->swap(G_SVD.U); /* V[i] now owns the data in U, so that when G_SVD
			  *  is destroyed, the data will still be there */

    for(size_t j=0;j<G_SVD.S.size();j++)
      evals(i,j) = pow( G_SVD.S(j),2 );

    Vec_data_t evri = evals.row(i);
    
    rho(i) = ( evri.SubVector(d_out,k).SumElements() ) / 
      ( evri.SubVector(0,d_out).SumElements() );
  }
  
  //find eta.  This is the median value of the rho array
  Vec_index_t indices;
  argsort(rho,indices);
  double eta = rho(indices(int(N/2)));

  if(verbose > 2)
    std::cout << "   + computing MLLE weight matrix\n";
  
  //Phi is our global weight matrix
  Matrix<double,SYM> Phi(N);
  for(size_t i=0; i<N; i++){
    //determine si - the size of the largest set of eigenvalues
    // of Qi such that satisfies:
    //    sum(in_set)/sum(not_in_set) < eta
    // with the constraint that 0<si<=k-d_out
    size_t si = 1;
    Vec_data_t evri = evals.row(i);
    while ( si<k-d_out ){
      double this_eta = evri.SubVector(k-si,k).SumElements() /
	evri.SubVector(0,k-si).SumElements();
      if(this_eta > eta){
	if(si!=1) --si;
	break;
      }else{
	++si;
      }
    }

    //select bottom si eigenvectors of Qi
    // and calculate alpha
    Mat_data_t Vi(k,si);
    for(size_t i1=0;i1<k;i1++)
      for(size_t i2=0;i2<si;i2++)
	Vi(i1,i2) = (*(V[i]))(i1,k-si+i2);

    Vec_data_t Vi_sum(si);
    for(size_t j=0;j<si;j++)
      Vi_sum(j) = Vi.col(j).SumElements();

    double alpha_i = blas_NRM2( Vi_sum )/sqrt(si);

    //compute Householder matrix which satisfies
    //  Hi*Vi.T*one(k) = alpha_i*one(s)
    // using proscription from paper
    Mat_data_t h (si,1,1.0);
    Mat_data_t one_k(k,1,1.0);
    h *= alpha_i;
    h -= (Vi.Transpose() * one_k);
    double nh = blas_NRM2(h);
    if(nh < TOL)
      h *= 0;
    else
      h /= nh;

    Matrix<double,SYM> Hi(si,0.0);
    Hi.diag() += 1.0;
    Matrix<double,SYM> hhT = h*h.Transpose();
    Hi -= 2*hhT;

    Mat_data_t Wi(k,si);
    Wi = Vi*Hi;
    for(size_t j=0;j<si;j++)
      Wi.col(j) += (1-alpha_i) * w_reg.col(i);

    Mat_data_t W_hat(N,si,0.0);
    for(size_t j=0; j<k; j++){
      int neighbor_j = neighbors(j,i);
      W_hat.row(neighbor_j) = Wi.row(j);
    }
    W_hat.row(i) -= 1;

    Phi += W_hat * W_hat.Transpose();
  }

  if(verbose > 1)
    std::cout << " - Finding null space of weight matrix with ARPACK\n";
  //now find eigenvectors of W...
  
  if(verbose > 2)
    std::cout << "   + ";

  EIGS_AR C_EIGS(Phi,d_out,"SM",1,!(verbose>2) );
  
  if(verbose > 1)
    std::cout << " - copying training proj\n";
  //copy values to training_proj_
  projection = C_EIGS.evecs.Transpose();
  
  if(verbose > 1)
    std::cout << " - Success!!\n";

  for(size_t i=0;i<V.size();i++)
    delete V[i];
  
}

void project_onto_LLE(const Mat_data_t& training_data,
		      const Mat_data_t& test_data,
		      int k,
		      Mat_data_t& test_proj,
		      int verbose){
  int d_out = test_proj.nrows();
;
  Mat_data_t training_proj(d_out,training_data.ncols());
  LLE(training_data,k,training_proj,verbose);

  Mat_index_t test_neighbors(k,test_data.ncols());
  compute_neighbors(training_data,test_data,test_neighbors,false,verbose);
    
  project_onto_LLE(training_data,
		   training_proj,
		   test_data,
		   test_neighbors,
		   test_proj,
		   verbose);
}

void project_onto_LLE(const Mat_data_t& training_data,
		      const Mat_data_t& training_proj,
		      const Mat_data_t& test_data,
		      int k,
		      Mat_data_t& test_proj,
		      int verbose){
  Mat_index_t test_neighbors(k,test_data.ncols());
  compute_neighbors(training_data,test_data,test_neighbors,false,verbose);

  project_onto_LLE(training_data,
		   training_proj,
		   test_data,
		   test_neighbors,
		   test_proj,
		   verbose);
}

void project_onto_LLE(const Mat_data_t& training_data,
		      const Mat_data_t& training_proj,
		      const Mat_data_t& test_data,
		      const Mat_index_t& test_neighbors,
		      Mat_data_t& test_proj,
		      int verbose){

  int N1 = training_data.ncols();
  int N2 = test_data.ncols();

  if(verbose>0)
    std::cout << "project_onto_LLE() : projecting new points onto previously computed projection\n";

  if( test_proj.ncols() != test_data.ncols() )
    throw LLE_Error("project_onto_LLE : test_proj must have same number of points as test_data:");

  if(test_proj.nrows() != training_proj.nrows() )
    throw LLE_Error("project_onto_LLE : test_proj and training_proj must be the same dimension");
  
  Mat_data_t W(N2,N1);
  compute_LLE_weights(training_data,test_data,test_neighbors,W,verbose);
  
  test_proj = training_proj * W.Transpose();
}

void compute_RLLE_scores(const Mat_data_t& training_data,
			 int k,
			 Vec_data_t& r_scores,
			 int d_out,
			 int verbose){
  Mat_index_t neighbors(k,training_data.ncols());
  compute_neighbors(training_data,neighbors,false,verbose);
  compute_RLLE_scores(training_data,neighbors,r_scores,d_out,verbose);
}

void compute_RLLE_scores(const Mat_data_t& training_data,
			 const Mat_index_t& neighbors,
			 Vec_data_t& r_scores,
			 int d_out,
			 int verbose){
  size_t N = training_data.ncols();
  size_t d_in = training_data.nrows();
  size_t k = neighbors.nrows();
  
  if(neighbors.ncols() != N)
    throw LLE_Error("compute_RLLE_scores : neighbors and training_data must have same ncols");
  
  if (r_scores.size() != N)
    throw LLE_Error("compute_RLLE_scores : r_scores and training_data must have same ncols");
  
  if (verbose > 1)
    std::cout << " - Determining reliability scores based on\n"
	      << "    Iteratively Reweighted PCA of each neighborhood\n";
    
  r_scores.SetAllTo(0.0);
    
  //data structures needed for finding scores
  Mat_data_t neighborhood(d_in,k);
    
  //data structures needed for IRWPCA procedure
  Mat_data_t trans(d_out,d_in);
  Vec_data_t mu(d_in);
  Vec_data_t A(k);

  for (size_t i=0;i<N;i++)
    {
      //construct neighborhood
      for(size_t j=0;j<k;j++){
	neighborhood.col(j) = 
	  training_data.col(neighbors(j,i)) - training_data.col(i);
      }
      //find weights with Iteratively Reweighted PCA of neighborhood
      try
	{
	  IRWPCA(neighborhood,A,mu,trans);
	}
      catch(IterException& ex)
	{
	  std::cerr << "   + IRWPCA did not converge for point "
		    << i << ": using equal weights in this neighborhood.\n";
	  A.SetAllTo(1.0/k);
	}
	
      //add weights to reliability scores
      for(size_t j=0;j<k;j++)
	r_scores(neighbors(j,i)) += A(j);
    }
}

//----------------------------------------------------------------------

void compute_weighted_neighbors(const Mat_data_t& training_data, 
				const Vec_data_t& r_scores,
				int r,
				Mat_index_t& neighbors,
				int verbose){
  size_t N = training_data.ncols();
  size_t d_in = training_data.nrows();
  size_t k = neighbors.nrows();
  if(neighbors.ncols() != N){
    throw LLE_Error("compute_weighted_neighbors : training_data and nbrs "
		      "must have same ncols");
  }
  if(r_scores.size() != N){
    throw LLE_Error("compute_weighted_neighbors : training_data and r_scores must have same ncols");
  }
  if(r<0)
    throw LLE_Error("compute_weighted_neighbors : r must be positive");

  if(k+r > N){
    throw LLE_Error("compute_weighted_neighbors : k+r must be less than "
		      "number of training points");
  }
  if(verbose>1) 
    std::cout << " - finding " << k << "+" << r << " nearest neighbors "
	      << "of each point\n"
	      << "    and reducing to " << k << " most reliable\n";
  
  Mat_data_t data_i(d_in,N);
  Vec_data_t D(N);
  Vec_index_t indices(N);
  
  for(size_t i=0;i<N;i++){
    //  find k+r nearest neighbors of point i
    // compute distances from point i to all other points
    for (size_t j=0;j<N;j++){
      data_i.col(j) = training_data.col(i) - training_data.col(j);
      D(j) = blas_NRM2( data_i.col(j) );
    }
    // find k+r nearest points
    argsort(D,indices);
    
    //now find the k of these with largest r_scores
    //swap indices until smallest r_scores are in positions k+1...k+r
    for(size_t j1=k+r; j1>k; j1--)
      for(size_t j2=1; j2<j1; j2++)
	if( r_scores(j1) > r_scores(j2) ) 
	  indices.swap(j1,j2);
    
    neighbors.col(i) = indices.SubVector(1,k+1);
  }
}

void RLLE1(const Mat_data_t& training_data,
	   int k,	
	   Mat_data_t& projection,
	   double r,
	   char proj_type,
	   int verbose){
  Mat_index_t neighbors(k,training_data.ncols());
  compute_neighbors(training_data,neighbors,false,verbose);
  RLLE1(training_data,neighbors,projection,r,proj_type,verbose);
}

void RLLE1(const Mat_data_t& training_data,
	   const Mat_index_t& neighbors,	
	   Mat_data_t& projection,
	   double r,
	   char proj_type,
	   int verbose){

  size_t N = training_data.ncols();
  size_t d_in = training_data.nrows();
  size_t k = neighbors.nrows();
  size_t d_out = projection.nrows();
  
  if(proj_type!='L' && proj_type!='H' && proj_type!='M')
    throw LLE_Error("RLLE1 : proj_type must be 'L','H', or 'M'");

  if(neighbors.ncols() != N)
    throw LLE_Error("RLLE1 : neighbors and training_data must have same ncols");
  
  if (projection.ncols() != N)
    throw LLE_Error("RLLE1 : projection and training_data must have same ncols");

  if(r<0 || r>=1)
    throw LLE_Error("RLLE1 : r must satisfy 0 <= r < 1");

  if (verbose > 0)
    std::cout << "RLLE1():\n";
  if(verbose > 1)
    std::cout << " - Performing RLLE on " << N <<" points in " 
	      << d_in << "->"<< d_out << " dimensions.\n";
  
  Vec_data_t r_scores(N);
  compute_RLLE_scores(training_data, neighbors, r_scores,
		      d_out, verbose);
  
  //------------------------------------------------
  // Determine how many points to cut
  int N_cut = int(r*N);
  
  Vec_index_t indices;
  argsort(r_scores,indices);
  double R_cutoff = r_scores(indices(N_cut));
  
  //------------------------------------------------
  // Construct a matrix of good data
  if (verbose>1)
    std::cout << " - Cutting " << N_cut << " out of " << N
	      << " points with reliability scores < " << R_cutoff << "\n";
  
  Mat_data_t data_trunc(d_in,N-N_cut);
  Mat_data_t proj_trunc(d_out,N-N_cut);
  for(size_t i=N_cut;i<N;i++){
    data_trunc.col(i-N_cut) = training_data.col(indices(i));
  }
  //------------------------------------------------
  // use LLE to obtain projection
  if (verbose > 2){
    std::cout << " - Using ";
    if(proj_type!='L') std::cout << proj_type;
    std::cout << "LLE to project truncated data\n";
  }

  if(proj_type=='H')
    HLLE(data_trunc,k,proj_trunc,verbose);
  else if(proj_type=='M')
    MLLE(data_trunc,k,proj_trunc,1E-12,verbose);
  else
    LLE(data_trunc,k,proj_trunc,verbose);

  project_onto_LLE(data_trunc,proj_trunc,
		   training_data,k,projection,verbose);
}

void RLLE2(const Mat_data_t& training_data,
	   int k,	
	   Mat_data_t& projection,
	   int r,
	   int verbose){
  Mat_index_t neighbors(k+r,training_data.ncols());
  compute_neighbors(training_data,neighbors,false,verbose);
  RLLE2(training_data,neighbors,projection,r,verbose);
}
			 
void RLLE2(const Mat_data_t& training_data,
	   const Mat_index_t& neighbors,	
	   Mat_data_t& projection,
	   int r,
	   int verbose){
  size_t N = training_data.ncols();
  size_t d_in = training_data.nrows();
  size_t k = neighbors.nrows();
  size_t d_out = projection.nrows();
  if(neighbors.ncols() != N){
    throw LLE_Error("RLLE2 : training_data and nbrs "
		      "must have same ncols");
  }
  if(projection.ncols() != N){
    throw LLE_Error("RLLE2 : training_data and projection must have same ncols");
  }
  if(r<0)
    throw LLE_Error("RLLE2 : r must be positive");

  if(k+r > N){
    throw LLE_Error("RLLE2 : k+r must be less than "
		      "number of training points");
  }
  if(d_out > d_in)
    throw LLE_Error("RLLE2 : must have d_out<=d_in");

  if (verbose > 0)
    std::cout << "RLLE2::compute_projection():\n"
	      << " - Performing RLLE on " << N <<" points in " 
	      << d_in << "->"<< d_out << " dimensions.\n";
  
  // Determine r_scores
  Vec_data_t r_scores(N);
  compute_RLLE_scores(training_data, neighbors, r_scores,
		      d_out, verbose);
  
  //-------------------------------------------------
  // Find k nearest neighbors to each point
  Mat_index_t weighted_neighbors(k,N);
  compute_weighted_neighbors(training_data, r_scores,
			     r, weighted_neighbors,verbose);
  
  //-------------------------------------------------
  // Find weight matrix
  Mat_data_t W(N,N);
  compute_LLE_weights(training_data,weighted_neighbors,W,verbose);
  
  //------------------------------------------------
  // Find null space to obtain projection
  if (verbose>1)
    std::cout << " - Finding r-score-weighted null space of weight matrix\n";
  if(verbose>2)
    std::cout << "    + constructing Covariance matrix\n";

  /* for LLE we use M = (I-W)^T * (I-W)
   *  and find the null space of M using the eigenvalue problem
   *   M * v = lam * v
   *
   * for RLLE2 we use the weights in R_scores
   *  let S = R_scores^2
   *  find the null space of (S*M) using the general eigenvalue problem
   *    M * v = lam * S^-1 * v
   */
  
  //let W = (W-I)
  W.diag() -= 1.0;
  Matrix<double,SYM> Cov = W.Transpose() * W;
  //clear weight matrix: it's no longer needed
  W.reallocate(0,0);
  
  Mat_data_t S(N,N,0.0);
  
  //let S.diag() = R^-2.  If R[i]==0, set it to 1E-15
  for(size_t i=0;i<N;i++)
    {
      if(r_scores(i)>0)
	S(i,i) = 1.0 / pow(r_scores(i),2);
      else
	S(i,i) = 1E30;
    }
  
  if (verbose>2)
    std::cout << "    + finding null space of Covariance Matrix with ARPACK\n"
	      << "    + ";
  
  EIGS_AR C_EIGS(Cov,d_out,"SM",1,!(verbose>2) );
  
  if (verbose>1)
    std::cout << " - Success!!\n";

  //copy values to training_proj_
  projection = C_EIGS.evecs.Transpose();

}

void compute_sigma(const Mat_data_t& training_data,
		   int k,
		   int d_out,
		   Vec_data_t& sigma,
		   int verbose/* = 0*/){
  Mat_index_t neighbors( k,training_data.ncols() );
  compute_neighbors(training_data,neighbors,false,verbose);
  compute_sigma(training_data,neighbors,d_out,
		sigma,verbose);
} 

void compute_sigma(const Mat_data_t& training_data,
		   const Mat_index_t& neighbors,
		   int d_out,
		   Vec_data_t& sigma,
		   int verbose/* = 0*/){
  unsigned int N = training_data.ncols();
  unsigned int d_in = training_data.nrows();
  unsigned int k = neighbors.nrows();

  if( neighbors.ncols() != N )
    throw LLE_Error("compute_sigma : training_data and neighbors "
		    "must have same ncols");

  if( d_out >= d_in )
    throw LLE_Error("compute_sigma : d_out must be less than d_in");

  if( sigma.size() != N )
    throw LLE_Error("sigma is the incorrect size");
  

  if (verbose > 0)
    std::cout << "compute_sigma:\n";
  
  if (verbose > 1)
    std::cout << " - Computing reconstruction errors of " 
	      << N <<" points in " << d_in << " dimensions.\n";
  
  Mat_data_t neighborhood(d_in,k);

  sigma.SetAllTo(0.0);
  
  //------------------------------------------------
  // Compute optimal reconstruction of each point from its neighborhood
  for (size_t i=0; i<N; i++)
    {
      for(int j=0;j<k;j++){
	neighborhood.col(j) = 
	  training_data.col( neighbors(j,i) ) - training_data.col(i);
      }
      
      //find singular values of neighborhood
      SVD nbrs_SVD(neighborhood,false);
      
      //find reconstruction error based on unused variance
      for(int j=d_out;j<std::min(d_in,k);j++)
	sigma(i) += nbrs_SVD.S(j) * nbrs_SVD.S(j);
    }
}
	       
void compute_mean_dist(const Mat_data_t& training_data,
		       Vec_data_t& mean_distances,
		       int verbose){
  if(training_data.ncols() != mean_distances.size()){
    throw LLE_Error("compute_mean_dist : training_data and mean_distances "
		      "must have same size");
  }

  //if test data and training data are the same, then the nearest
  // neighbor of a point is itself.  Use a variable starting_k to
  // correct for this
  
  int D = training_data.nrows();
  int N = training_data.ncols();
  
  if(verbose>0)
    std::cout << "compute_mean_dist() : computing mean distance of "
	      << N << " points in " << D << " dimensions\n";
  
  //temporary vectors for arithemetic
  Vec_data_t diff_ij(D);
  
  for(int i=0;i<N;i++)
    {
      double d_sum = 0;
      //compute distance from point i to all other points
      for (int j=0;j<N;j++){
	diff_ij = training_data.col(i) - training_data.col(j);
	d_sum += sqrt( blas_NRM2( diff_ij ) );
      }
      mean_distances(i) = d_sum/N;
    }
}
