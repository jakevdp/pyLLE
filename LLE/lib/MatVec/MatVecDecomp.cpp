#include "MatVec.h"
#include "lapack.h"
#include "arpack.h"
#include "MatVecDecomp.h"

SVD::SVD(Matrix<double,DENSE>& A, 
	 bool compute_uv/*=true*/,
	 bool overwrite/*=false*/,
	 bool full_matrices/*=false*/){
  if(compute_uv && full_matrices && overwrite){
    std::cerr << "Warning: SVD: cannot overwrite full matrices. "
	      << "Computing only partial matrices\n";
    full_matrices = false;
  }

  bool switch_uv = false;
  if(A.isTransposed() ){
    //throw MatVecError("SVD : not implemented for Transposed Matrices");
    switch_uv = true;
    A.TransposeSelf();
  }
  
  char JOBZ;
  int M = A.nrows();
  int N = A.ncols();
  int K = A.rank();
  
  //in the LAPACK routine, the input matrix A is destroyed
  // so declare A_alg, the copy or view of A that will be 
  // used for the algorithm
  Matrix<double,DENSE> A_alg;
  
  if(compute_uv){
    //compute partial matrices & overwrite A
    if(overwrite){
      JOBZ = 'O';
      if(full_matrices){
	std::cerr << "Warning: SVD: cannot overwrite full matrices. "
		  << "Computing only partial matrices\n";
      }
      if(M>=N){
	VT.reallocate(N,N);
	U.viewof(A);
      }else{
	U.reallocate(M,M);
	VT.viewof(A);
      }
      A_alg.viewof(A);
    }
    //compute full matrices
    else if(full_matrices){
      JOBZ = 'A';
      U.reallocate(M,M);
      VT.reallocate(N,N);
      A_alg.deep_copy(A);
    }
    //compute partial matrices
    else{
      JOBZ = 'S';
      U.reallocate(M,K);
      VT.reallocate(K,N);
      A_alg.deep_copy(A);
    }
  }
  //do not compute U or V
  else{
    JOBZ = 'N';
    A_alg.deep_copy(A);
  }
  
  int LDU = (U.ld()>1) ? U.ld() : 1;
  int LDVT = (VT.ld()>1) ? VT.ld() : 1;
  S.reallocate(K);
  
  dgesdd( JOBZ, M, N, A_alg.arr(), A_alg.ld(), S.arr(), 
	  U.arr(), LDU, VT.arr(), LDVT );
  
  if(switch_uv){
    A.TransposeSelf();
    if(compute_uv){
      U.swap(VT);
      U.TransposeSelf();
      VT.TransposeSelf();
    }
  }
}
  


LUD::LUD(const Matrix <double,DENSE>& A) 
  : LU(A), IPIV(A.rank()), flag_(A.isTransposed()), sym_(false){
  dgetrf(A.nrows(), A.ncols(), LU.arr(), A.ld(), IPIV.arr() );
}

LUD::LUD(const Matrix <double,SYM>& A) 
  : LU(A.nrows(),A.nrows()), IPIV(A.rank()), flag_(A.upper()), sym_(true){
  //copy matrix from A to LU
  dcopy(A.size(),A.arr(),A.inc(),LU.arr(),LU.inc() );
  
  dsytrf(A.upper(), LU.nrows(), LU.arr(), LU.ld(), IPIV.arr() );
}
  


QRD::QRD(Matrix <double,DENSE>& A) 
  : Q(A.nrows(),A.rank()), R(A.rank(),A.ncols(),0.0){
  if(A.isTransposed() )
    throw NotImplementedError("QRD of transposed matrix");

  int K = A.nrows();

  Matrix<double> A_alg;
  if(A.nrows() < A.ncols()){
    A_alg.deep_copy(A);
  }else{
    Q = A;
    A_alg.viewof(Q);
  }
  
  double* TAU = new double[A.rank()];
  
  dgeqrf(A_alg.nrows(), A_alg.ncols(), A_alg.arr(), A_alg.ld(), TAU);

  //copy R
  for(int j=0;j<A.ncols();j++)
    for(int i=0;i<=std::min(j,K-1);i++)
      R(i,j) = A_alg(i,j);
  
  dorgqr(A_alg.nrows(), A_alg.rank(), A_alg.rank(), 
	 A_alg.arr(), A_alg.ld(), TAU);
  
  //copy Q
  if(A.nrows() < A.ncols()){
    for(int i=0;i<A.nrows();i++)
      for(int j=0;j<A.nrows();j++)
	Q(i,j) = A_alg(i,j);
  }
  
      
  delete[] TAU;
}

/************************************************************/


EIGS_AR::EIGS_AR(Matrix<double,SYM>& A, int nev, 
		 const std::string& which/*="LA"*/,
		 int num_to_ignore/*=0*/, bool quiet/*=false*/)
  : evecs(A.nrows(),nev), evals(nev), quiet_(quiet){
  Matrix<double,SYM> B(1,1.0);
  compute_eigs(A,B,which,num_to_ignore,false);
}
  
EIGS_AR::EIGS_AR(Matrix<double,SYM>& A, const Matrix<double,SYM>& B, 
		 int nev, const std::string& which/*="LA"*/,
		 int num_to_ignore/*=0*/, bool quiet/*=false*/)
  : evecs(A.nrows(),nev), evals(nev), quiet_(quiet){
  compute_eigs(A,B,which,num_to_ignore,true);
}


void EIGS_AR::compute_eigs(Matrix<double,SYM>& A, 
			   const Matrix<double,SYM>& B, 
			   const std::string& which,
			   int num_to_ignore, bool general){
  int Mode; /* Sets the mode of dsaupd.
	       1 is exact shifting,
	       2 is user-supplied shifts,
	       3 is shift-invert mode,
	       4 is buckling mode,
	       5 is Cayley mode. */
  if(which == std::string("SM") )
    Mode = 3;
  else if(which == std::string("LM") )
    Mode = general ? 2 : 1;
  else
    throw MatVecError("EIGS_AR : only SM and LM are supported");

  int N = A.nrows();
  int Max_iter = 30;//3*N;
  int NEV = evals.size();

  if( NEV >= A.nrows() )
    throw MatVecError("EIGS_AR: nev must be less than dimension of matrix");
    
  if(general && ((B.nrows() != N) || (B.ncols() != N)) )
    throw MatVecError("EIGS_AR: B must be the same size as A for "
		      "general problem\n");
    
  /* find LU decomposition of A for shift-invert mode
     or of B for normal mode*/
  LUD AB_LU((Mode==3) ? A : B);
    
  /* Initialize arguments for ARPACK routine dsaupd */
  NEV += num_to_ignore;
    
  /* IDO is the reverse communication flag */
  int IDO = 0; 
  
  /* BMAT specifies whether this is a standard or
     generalized eigenvalue problem.  
     'I' means standard 
     'G' means general  */
  char BMAT = general ? 'G':'I';
  
  /* WHICH specifies which ritz values to compute.  Because
     we're using shift-invert mode and want to find the
     eigenvalues closest to sigma=0, we want the largest 
     magnitude ("LM") eigenvalues of (A-sigma*I)^-1 */
  char* WHICH = "LM";
  
  /* TOL gives the convergence precision.
     TOL<=0 means use machine precision */
  double TOL = 0.0;
  
  /* RESID is the initial residual vector.  If info===0 on
     input, this is an initial guess.  On output this contains
     the final residual vector */
  double *RESID = new double[N];
  
  /* V is a matrix of size [LDV x NCV].
     LDV is the leading dimension of V: equal to N in our case.
     NCV gives the number of columns of v, which holds the basis
     vectors computed in the Implicitly Restarted Arnoldi
     Process.  NCV must be >= 2*NEV, and <= N.
     Note that the cost per iteration is N*NCV*NCV. */
  int NCV = (4*NEV < N) ? 4*NEV : N;
  int LDV = N;
  double *V = new double[LDV*NCV];
  
  /* IPARAM is an array that is used to pass information
     to the algorithm */
  int *IPARAM = new int[11];
  
  IPARAM[0] = 1;// Specifies the shift strategy (1 is automatic shifts)
  IPARAM[2] = Max_iter;// Maximum number of iterations
  IPARAM[3] = 1; //blocksize: must be 1
  IPARAM[6] = Mode; 
  
  /* IPNTR indicates the locations in the work array WORKD
     where the input and output vectors in the
     callback routine are located. */
  int *IPNTR = new int[11];
  
  /* WORKD holds the vectors for the reverse communication interface */
  double *WORKD = new double[3*N];
  
  /* WORKL is an array of length LWORKL, and provides workspace
     for the algorithm (must have LWORKL >= NCV*(NCV+8) */
  int LWORKL = NCV*(NCV+8);
  double *WORKL = new double[LWORKL];
  
  /* INFO passes convergence information out of the iteration
     routine. */
  int INFO = 0;
  
  /* call the ARPACK dsaupd routine.  ido parameter tells us
     about the convergence. */
  do{
    dsaupd_(&IDO, &BMAT, &N, WHICH, &NEV, &TOL, RESID, 
	    &NCV, V, &LDV, IPARAM, IPNTR, WORKD, WORKL,
	    &LWORKL, &INFO);
    
    Vector<double,GEN> iptr1(N, WORKD+(IPNTR[0]-1));
    Vector<double,GEN> iptr2(N, WORKD+(IPNTR[1]-1));
    Vector<double,GEN> iptr3(N, WORKD+(IPNTR[2]-1));
    
    switch(IDO)
      {
      case -1: 
      case 1:
	if(Mode==3){
	  //compute iptr2 = A.inv * B * iptr1
	  if(IDO==1)
	    iptr2 = iptr3; //B*iptr1 is stored in iptr3
	  else if(general)
	    iptr2 = B*iptr1;
	  else
	    iptr2 = iptr1;
	  SOLVE(AB_LU, iptr2);
	}else{
	  //compute iptr2 = B.inv * A * iptr1
	  iptr2 = A*iptr1;
	  if(general)
	    SOLVE(AB_LU, iptr2);
	}
	break;
	  
      case 2:
	//compute iptr2 = B * iptr1 
	if(general)
	  iptr2 = B*iptr1;
	else
	  iptr2 = iptr1;
	
	break;
	
      case 3:
	//should not happen : not using shifts
	std::cerr << "Warning: IDO=3 reached\n";
      }
  }while(IDO!=99);
    
  /* Now extract the eigenvalues and eigenvectors using the
     ARPACK dseupd routine. */
  
  if (INFO<0) 
    {
      std::cerr << "EIGS_AR: Error with dsaupd, INFO = " << INFO << "\n";
      std::cerr << "Check documentation in dsaupd\n\n";
      if(INFO==-9999){
	std::cerr << "  IPARAM[5] = " << IPARAM[4] << '\n'
		  << "  Note: this could be an ARPACK linking problem.\n"
		  << "   see http://mathema.tician.de/node/373\n";
      }
    } 
  
  /* Initialize parameters needed for ARPACK routine dseupd */
    
  /* RVEC specifies whether eigenvectors should be calculated 
     1 -> should be
     0 -> should not be*/
  int RVEC = 1;
  
  /* HOWMNY specifies which vectors to calculate. 
       currently, only 'A' is allowed */
  char HOWMNY = 'A';
  
  /* SELECT specifies the specific vectors to calculate if HOWMNY=="S" 
      should be unreferenced according to the documentation, but unless
      it's a valid array, we get a seg-fault.  Not sure why...*/
  int* SELECT = new int[NCV];

  /* D will contain the eigenvalues upon output */
  double *D = new double[2*NCV]; 
  
  /* SIGMA represents the shift */
  double SIGMA = 0.0;
  
  /* IERR gives return information */
  int IERR;
  
  dseupd_(&RVEC, &HOWMNY, SELECT, D, V, &LDV, &SIGMA, &BMAT,
	  &N, WHICH, &NEV, &TOL, RESID, &NCV, V, &LDV,
	  IPARAM, IPNTR, WORKD, WORKL, &LWORKL, &IERR);
  if (IERR!=0) 
    {
      std::cerr << "EIGS_AR: Error with dseupd, INFO = " << IERR << "\n";
      std::cerr << "Check the documentation of dseupd.\n";
    } 
  else if (INFO==1) 
    {
      std::cerr << "EIGS_AR: Maximum number of iterations reached: " 
		<< IPARAM[2] << "\n";
    } 
  else if (INFO==3) 
    {
      std::cerr << "EIGS_AR: No shifts could be applied during implicit\n";
      std::cerr << "Arnoldi update, try increasing NCV.\n";
    }
  else if(!quiet_)
    {
      std::cout << "EIGS_AR: Converged in " << IPARAM[2] << " iterations.\n";
    }
  
  /* Copy eigenvalues and eigenvectors */ 
  dcopy(evals.size(),D + num_to_ignore,1,evals.arr(),1); 
  dcopy(evecs.size(),V+N*num_to_ignore,1,evecs.arr(),1 );
  
  /* deallocate memory */
  delete[] RESID;
  delete[] V;
  delete[] IPARAM;
  delete[] IPNTR;
  delete[] WORKL;
  delete[] WORKD;
  delete[] SELECT;
  delete[] D;
};



EIGS::EIGS(Matrix<double,SYM>& A, bool compute_v)
  : evals(A.nrows()), evecs(A.nrows(),A.ncols()){
  dcopy(A.size(),A.arr(),A.inc(),evecs.arr(),evecs.inc());

  dsyev(compute_v, A.upper(), A.nrows(), evecs.arr(), evecs.ld(), evals.arr());
  
  if(!compute_v) evecs.reallocate(0,0);
}


/************************************************************/

template<>
void SOLVE(LUD& A, 
	    Matrix<double,DENSE>& X,
	    const Matrix<double,DENSE>& B){
  if(A.LU.nrows()!=A.LU.ncols())
    throw MatVecError("SOLVE works only for sqare A\n");
  X = B;
  SOLVE(A,X);
}


template<>
void SOLVE(LUD& A, 
	    Matrix<double,DENSE>& BX){
  if(A.LU.nrows()!=A.LU.ncols())
    throw MatVecError("SOLVE works only for sqare A\n");
  
  if( A.sym() )
    dsytrs(A.flag(), A.LU.nrows(), BX.ncols(), A.LU.arr(), 
	   A.LU.ld(), A.IPIV.arr(), BX.arr(), BX.ld());
  else
    dgetrs(A.flag(), A.LU.nrows(), BX.ncols(), A.LU.arr(), 
	   A.LU.ld(), A.IPIV.arr(), BX.arr(), BX.ld());
}


template<>
void SOLVE(LUD& A, 
	   Vector<double,GEN>& x,
	   const Vector<double,GEN>& b){
  if(A.LU.nrows()!=A.LU.ncols())
    throw MatVecError("SOLVE works only for sqare A\n");
  x = b;
  if(x.inc() != 1)
    throw MatVecError("SOLVE works only for inc=1 vectors\n");
  SOLVE(A,x);
}

template<>
void SOLVE(LUD& A, 
	   Vector<double,GEN>& bx){
  if(A.LU.nrows()!=A.LU.ncols())
    throw MatVecError("SOLVE works only for sqare A\n");

  if(bx.inc() != 1)
    throw MatVecError("SOLVE works only for inc=1 vectors\n");

  if( A.sym() )
    dsytrs(A.flag(), A.LU.nrows(), 1, A.LU.arr(), 
	   A.LU.ld(), A.IPIV.arr(), bx.arr(), bx.size());
  else
    dgetrs(A.flag(), A.LU.nrows(), 1, A.LU.arr(), 
	   A.LU.ld(), A.IPIV.arr(), bx.arr(), bx.size());
}
