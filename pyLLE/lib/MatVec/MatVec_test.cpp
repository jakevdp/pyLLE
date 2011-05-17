#include "MatVec.h"
#include "MatSym.h"
#include "MatTri.h"
#include "MatVecDecomp.h"
#include <time.h>
#include <math.h>
#include <iostream>
#include <stdlib.h> //for rand() and srand()


using namespace std;

void initrand(Matrix<double>& M){
  srand(time(0));
  for(size_t i=0;i<M.nrows();i++)
    for(size_t j=0;j<M.ncols();j++)
      M(i,j) = 0.001 * ( rand() - RAND_MAX/2);
  
}

void initmatrix(Matrix<double>& M){
  for(size_t i=0;i<M.nrows();i++)
    for(size_t j=0;j<M.ncols();j++)
      M(i,j) = i+2*j+0.5 + (i<j?1:0);
}

void initsymmatrix(Matrix<double,SYM>& M){
  for(size_t i=0;i<M.nrows();i++){
    for(size_t j=0;j<M.ncols();j++){
      M(i,j) = 1+2*(i+j) + i*std::min(i,j);
    }
  }
}

void initsymmatrix(Matrix<double,DENSE>& M){
  for(size_t i=0;i<M.nrows();i++)
    for(size_t j=0;j<M.ncols();j++)
      M(i,j) = M(j,i) = 1+2*(i+j) + std::min(i,j);
}

void initmatrix_B(Matrix<double>& M){
  for(size_t i=0;i<M.nrows();i++)
    for(size_t j=0;j<M.ncols();j++)
      M(i,j) = 0.5*i+j + (i<j?1:0.5);
}

void initvector(Vector<double>& V){
  for(size_t i=0;i<V.size();i++)
    V(i)= i+1;
}

void test_Transpose(){
  cout << "testing Transpose()\n";
  int Nrows=3;
  int Ncols=4;

  Matrix<double> A(Nrows,Ncols);
  initmatrix(A);

  std::cout << "A: " << A << "A.Transpose(): " << A.Transpose() << "\n";
}
  

void test_DCOPY_M(){
  cout << "testing DCOPY on Matrices\n";
  int Nrows=3;
  int Ncols=4;

  Matrix<double> A(Nrows,Ncols);
  initmatrix(A);
  
  Matrix<double> B(Nrows,Ncols);

  cout << "A: " << A << "B: " << B;

  cout << "\ncopying A to B:\n";

  B = A;
  cout << "A: " << A << "B: " << B;
}

void test_construct(){
  cout << "testing Matrix constructor:\n"
       << "  B and Bsym will be constructed from A*A^T\n";
  int Nrows=3;
  int Ncols=4;

  Matrix<double> A(Nrows,Ncols);
  initmatrix(A);
  
  Matrix<double> B(4.0*( A*A.Transpose() ) );
  Matrix<double,SYM> Bsym(4.0*( A*A.Transpose() ) );

  cout << "A: " << A << "B: " << B << "Bsym: " << Bsym;
}

void test_DCOPY_V(){
  cout << "testing DCOPY on Vectors\n";
  int N=10;

  Vector<double> x(N);
  initvector(x);

  Vector<double> y(N);

  cout << "x: " << x << "y: " << y;

  cout << "\ncopying x to y:\n";

  y = x;
  cout << "x: " << x << "y: " << y;
}

void test_LU_M(){
  cout << "testing LU decomposition on Matrices\n";
  int N_A = 5;

  //Matrix<double> A(N_A,N_A);
  //initmatrix(A);

  Matrix<double,SYM> A(N_A,N_A);
  initsymmatrix(A);

  int Nrows_B = N_A;
  int Ncols_B = 4;
  
  Matrix<double> B(Nrows_B,Ncols_B);
  initmatrix_B(B);

  Matrix<double> C(Nrows_B,Ncols_B);

  cout << "A: " << A << '\n';
  cout << "B: " << B << '\n';

  SOLVE(A,B);

  cout << "X = A^-1 * B: " << B << '\n';

  C = A*B;

  cout << "A*X: " << C << '\n';
}

void test_LU_V(){
  cout << "testing LU decomposition on Vector\n";
  int N_A = 5;

  int size_B = N_A;
  
  Matrix<double,SYM> A(N_A,N_A);
  initsymmatrix(A);
  
  Vector<double> B(size_B);
  initvector(B);

  Vector<double> C(size_B);

  cout << "A: " << A << '\n';
  cout << "b: " << B << '\n';

  SOLVE(A,B);

  cout << "x = A^-1 * b: " << B << '\n';

  C = A*B;

  cout << "A*x: " << C << '\n';
}


void test_DGEMM(){
  cout << "testing DGEMM\n";
  int Nrows_A = 5;
  int Ncols_A = 2;
  
  int Ncols_B = 3;
  int Nrows_B = Ncols_A;

  int Nrows_C = Nrows_A;
  int Ncols_C = Ncols_B;
  
  Matrix<double> A(Nrows_A,Ncols_A);
  initmatrix(A);
  
  Matrix<double> B(Nrows_B,Ncols_B);
  initmatrix_B(B);

  Matrix<double> C(Nrows_C,Ncols_C);

  C = A*B;
  
  cout<< "A: " << A << '\n';
  cout<< "B: " << B << '\n';
  cout<< "A*B " << C << '\n';
}

void test_DGEMV(){
  cout << "testing DGEMV\n";
  int Nrows_A = 5;
  int Ncols_A = 4;
  
  int size_y = Nrows_A;
  int size_x = Ncols_A;
  
  Matrix<double> A(Nrows_A,Ncols_A);
  initmatrix(A);

  Vector<double> x(size_x);
  initvector(x);

  Vector<double> y(size_y);
    

  y = A*x;
  
  cout<< "A: " << A << '\n';
  cout<< "x: " << x << '\n';
  cout<< "A*x: " << y << '\n';
}

void test_DSYMV(){
  cout << "testing DSYMV\n";
  int Nrows_A = 5;
  
  int size_y = Nrows_A;
  int size_x = Nrows_A;
  
  Matrix<double,SYM> A(Nrows_A);
  initsymmatrix(A);

  Vector<double> x(size_x);
  initvector(x);

  Vector<double> y(size_y);
  
  y = A*x;;
  
  cout<< "A: " << A << '\n';
  cout<< "x: " << x << '\n';
  cout<< "A*x: " << y << '\n';
}

void test_colrow(){
  cout << "testing cols and rows\n";
  int Nrows_A = 5;
  int Ncols_A = 4;
  
  Matrix<double> A(Nrows_A,Ncols_A);
  initmatrix(A);

  cout <<A;
  
  cout << "\nA.row(0):\n";
  cout <<A.row(0);
  cout << "\nA.row(1):\n";
  cout <<A.row(1);

  cout << "\nA.col(0):\n";
  cout << A.col(0);
  cout << "\nA.col(1):\n";
  cout << A.col(1);

  Vector<double> Acol1 = A.col(1);

  cout << "A.col(1)(1) = 100\n";
  Acol1(1) = 100;
  cout << Acol1;
  cout << A;

  cout << "\nA.diag():\n";
  cout << A.diag();

}

void test_DAXPY_V(){
  cout << "testing DAXPY on Vectors\n";
  int N = 10;

  Vector<double> V1(N);
  Vector<double> V2(N);
  initvector(V1);
  initvector(V2);

  cout << "V1: " << V1;
  cout << "V2: " << V2;

  V1+=V2;
  cout << "V1+V2: " << V1 << '\n';
}


void test_DAXPY_M(){
  cout << "testing DAXPY on Matrices\n";
  int Nrows = 3;
  int Ncols = 4;

  Matrix<double> M1(Nrows,Ncols);
  Matrix<double> M2(Nrows,Ncols);
  initmatrix(M1);
  initmatrix_B(M2);

  cout << "M1: " << M1;
  cout << "M2: " << M2;

  M1+=M2;
  cout << "M1+M2: " << M1 << '\n';
}

void test_SVD(){
  cout << "----------------------------------------------------------------------\n";
  cout << "testing SVD\n";
  int Nrows = 5;
  int Ncols = 4;
  int K = min(Nrows,Ncols);
  
  
  {
    Matrix<double> M(Nrows,Ncols);
    initmatrix(M);
    cout << "M: " << M;
    
    SVD M_SVD(M,true); //compute U & V
    
    cout << "U: " << M_SVD.U;
    cout << "S: " << M_SVD.S;
    cout << "VT: " << M_SVD.VT;
    
    Matrix<double> S_mat(K,K,0.0);
    
    for(size_t i=0; i<K; i++)
      S_mat(i,i) = M_SVD.S(i);
    
    Matrix<double> temp(K,Ncols);
    
    temp = S_mat*M_SVD.VT;
    M = M_SVD.U*temp;
    
    cout << "U*S*VT: " << M;

    SVD M_SVD2(M,false); //don't compute U & V
    cout << "S only: " << M_SVD2.S;
    cout << "U: " << M_SVD2.U;
    cout << "VT: " << M_SVD2.VT;
  }
  {
    
    cout << "----------------------------------------------------------------------\n";
    cout << "Testing with transpose:\n";
    
    Matrix<double> M(Ncols,Nrows); //transposed
    M.TransposeSelf();
    initmatrix(M);
    cout << "M: " << M;
    SVD M_SVD(M,true); //compute U & V
    
    cout << "U: " << M_SVD.U;
    cout << "S: " << M_SVD.S;
    cout << "VT: " << M_SVD.VT;
    
    Matrix<double> S_mat(K,K,0.0);
    
    for(size_t i=0; i<K; i++)
      S_mat(i,i) = M_SVD.S(i);
    
    Matrix<double> temp(K,Ncols);
    
    temp = S_mat*M_SVD.VT;
    M = M_SVD.U*temp;
    
    cout << "U*S*VT: " << M;
    
    SVD M_SVD2(M,false); //don't compute U & V
    cout << "S only: " << M_SVD2.S;
    cout << "U: " << M_SVD2.U;
    cout << "VT: " << M_SVD2.VT;
  }
}


void test_SubVector(){
  cout << "testing SubVector\n";
  int N = 6;

  Matrix<double> M(N,N);
  initmatrix(M);

  Vector<double> V = M.row(0);

  cout << "V: " << V;
  cout << "V.SubVector(0,3): " << V.SubVector(0,3);
  cout << "V.SubVector(3,6): " << V.SubVector(3,6);
}

void test_Trace(){
  cout << "testing Trace\n";
  int N = 6;

  Matrix<double> M(N,N);
  initmatrix(M);

  Vector<double> V = M.row(0);

  cout << "M: " << M;
  cout << "M.diag(): " << M.diag();
  cout << "M.Trace(): " << M.Trace() << "\n";
}

void test_DSCAL(){
  cout << "testing DSCAL\n";
  int N = 10;
  int Nrows = 4;
  int Ncols = 4;

  Vector<double> V(N);
  initvector(V);

  Matrix<double> M(Nrows,Ncols);
  initmatrix(M);

  cout << "V: " << V;
  V *= 2.0;
  cout << "V*2.0: " << V << "\n";

  cout << "M: " << M;
  M *= 2.0;
  cout << "M*2.0: " << M << "\n"; 
}

void test_QR(){
  cout << "testing QR decomposition\n";
  int Nrows = 4;
  int Ncols = 6;
  
  Matrix<double> M(Nrows,Ncols);
  initmatrix(M);

  cout << "M: " << M << "\n";
  
  QRD M_QRD(M);

  cout << "Q: " << M_QRD.Q;
  cout << "R: " << M_QRD.R << "\n";

  M = M_QRD.Q*M_QRD.R;

  cout << "Q*R: " << M;
  
  Matrix<double,SYM> Cov(M_QRD.Q.ncols());
  Cov = M_QRD.Q.Transpose() * M_QRD.Q;

  cout << "Q^T*Q: " << Cov;
}

void test_Tri(){
  std::cout << "testing triangular matrix\n";
  int Nrows = 5;
  int Ncols = 4;

  Matrix<double,TRI> M(Nrows,Ncols);

  for(size_t i=0;i<Nrows;i++){
    for(size_t j=i;j<Ncols;j++){
      M(i,j) = i+j + 1;
    }
  }

  std::cout << M << "\n";

  M.flip_unitdiag();

  std::cout << M << "\n";

  std::cout << M.Transpose() << "\n";
}

void test_EIGS(){
  std::cout << "testing EIGS_AR (arpack interface), looking for "
	    << "smallest eigenvalues/vectors\n";
  int Nrows = 5;
  int Ncols = 4;

  int d_out = 3;

  Matrix<double> M(Nrows,Ncols);
  initmatrix(M);
  Matrix<double,SYM> C = M * M.Transpose();

  std::cout << "Matrix C: " << C << std::endl;
  
  
  /*-------- standard problem -----------------*/

  std::cout << "\n-----------------------------------\n"
	    << "   Standard Problem\n\n";
  
  //ARPACK eigenvalues/vectors
  EIGS_AR C_EIGS_AR(C,3);

  //SVD eigenvalues/vectors
  EIGS C_EIGS(C);

  std::cout << "AREIG eigenvalues: " << C_EIGS_AR.evals << "\n"
	    << "SVD eigenvalues: " << C_EIGS.evals << "\n\n"
	    << "AREIG eigenvectors: " << C_EIGS_AR.evecs << "\n"
	    << "SVD eigenvectors: "
	    << C_EIGS.evecs << "\n"
	    << "\n-----------------------------------\n\n";
}

void test_EIGS_2(){
  std::cout << "testing EIGS_AR (arpack interface), looking for "
	    << "largest eigenvalues/vectors\n";
  int Nrows = 5;
  int Ncols = 4;

  int d_out = 3;

  Matrix<double> M(Nrows,Ncols);
  initmatrix(M);
  Matrix<double,SYM> C = M * M.Transpose();

  std::cout << "Matrix C: " << C << std::endl;
  
  
  /*-------- standard problem -----------------*/

  std::cout << "\n-----------------------------------\n"
	    << "   Standard Problem\n\n";
  
  //ARPACK eigenvalues/vectors
  EIGS_AR C_EIGS_AR(C,3,"LM");

  //SVD eigenvalues/vectors
  EIGS C_EIGS(C);

  std::cout << "AREIG eigenvalues: " << C_EIGS_AR.evals << "\n"
	    << "SVD eigenvalues " << C_EIGS.evals << "\n\n"
	    << "AREIG eigenvectors: " << C_EIGS_AR.evecs << "\n"
	    << "SVD eigenvectors: "
	    << C_EIGS.evecs << "\n"
	    << "\n-----------------------------------\n\n";
}
  

void test_const(){
  Matrix<double> M(4,5);
  initmatrix(M);
  Matrix<double> A = M;
  A.Transpose()(2,3) = 100;
  std::cout << "Set a view of (M.T)(2,3) = 100\n";
  std::cout << M;
}

void test_largeEIGS(){
  Matrix<double,SYM> M(1000);
  initsymmatrix(M);
  
  //EIGS E(M);
  EIGS_AR EAR(M,3,"LM");
}

void test_addscalar(){
  std::cout << "testing adding a scalar to a vector:\n";
  Matrix<double> Z(4,4);
  Z.diag() -= 1.0;
  std::cout << Z;

  Z(2,2) = 100;

  std::cout << Z;

  Z.diag().swap(1,2);

  std::cout << Z;
}

void test_eigspeed(int D=1000,int N=9000){
  Matrix<double> M(D,N);
  initrand(M);

  time_t t1;
  time_t t2;
  
  int D_OUT = 5;
  
  {
    std::cout << "computing SVD...\n";
    t1 = time(0);
    SVD M_SVD(M);
    t2 = time(0);
    for(int i=D_OUT-1;i>=0;i--)
      std::cout << M_SVD.S(i) * M_SVD.S(i) << " ";
    std::cout << "\n" << M_SVD.U.col(0).SubVector(0,10);
    std::cout << " time: " << difftime(t2,t1) << " sec\n";
  }
  {
    std::cout << "computing EIGS...\n";
    t1 = time(0);
    Matrix<double,SYM> C = M*M.Transpose();
    EIGS C_EIGS(C);
    Matrix<double> sing_vals(D_OUT,D,0.0);
    for(int i=0;i<D_OUT;i++)
      sing_vals(i,i) = sqrt(C_EIGS.evals(i));
    Matrix<double> tmp = sing_vals * C_EIGS.evecs.Transpose();
    Matrix<double> proj = tmp * M;
    t2 = time(0);
    std::cout << C_EIGS.evals.SubVector(D-D_OUT,D)
	      << C_EIGS.evecs.col(D-1).SubVector(0,10)
	      << " time: " << difftime(t2,t1) << " sec\n";
  }
  {
    std::cout << "computing EIGS_AR...\n";
    t1 = time(0);
    Matrix<double,SYM> C = M*M.Transpose();
    EIGS_AR C_EIGS(C,D_OUT,"LM");
    Matrix<double> sing_vals(D_OUT,D_OUT,0.0);
    for(int i=0;i<D_OUT;i++)
      sing_vals(i,i) = sqrt(C_EIGS.evals(i));
    Matrix<double> tmp = sing_vals * C_EIGS.evecs.Transpose();
    Matrix<double> proj = tmp * M;
    t2 = time(0);
    std::cout << C_EIGS.evals
	      << C_EIGS.evecs.col(D_OUT-1).SubVector(0,10)
	      << " time: " << difftime(t2,t1) << " sec\n";
  }
}


int main(int argc,char *argv[]){
  char arg = 'N';
  if(argc>1){
    arg = argv[1][0];
  }
  switch(arg)
    {
    case 'a': test_DCOPY_M();
      break;
    case 'b': test_DCOPY_V();
      break;
    case 'c': test_LU_V();
      break;
    case 'd': test_LU_M();
      break;
    case 'e': test_DGEMM();
      break;
    case 'f': test_DGEMV();
      break;
    case 'g': test_colrow();
      break;
    case 'h': test_DAXPY_V();
      break;
    case 'i': test_DAXPY_M();
      break;
    case 'j': test_SVD();
      break;
    case 'k': test_SubVector();
      break;
    case 'l': test_Trace();
      break;
    case 'm': test_DSCAL();
      break;
    case 'n': test_QR();
      break;
    case 'o': test_DSYMV();
      break;
    case 'p': test_Tri();
      break;
    case 'q': test_EIGS();
      break;
    case 'r': test_EIGS_2();
      break;
    case 's': test_construct();
      break;
    case 't': test_const();
      break;
    case 'u': test_addscalar();
      break;
    case 'v': test_largeEIGS();
      break;
    case 'w': test_eigspeed();
      break;
    case 'x': test_Transpose();
      break;
      
    default:
      cerr << "usage: MatVec_test [a-w]\n";
    }
  return 0;
}
