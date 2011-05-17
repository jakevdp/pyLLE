//#define DEBUG

#include "MatVec.h"
#include "BallTree.h"
#include "argsort.h"
#include <math.h>

#include <stdlib.h>
#include <time.h>

void compute_neighbors(const Matrix<double>& M,
		       const Vector<double>& pt,
		       Vector<int>& nbrs){
  int D = M.nrows();
  int N = M.ncols();
  int k = nbrs.size();
  if(pt.size() != D)
    throw MatVecError("compute_neighbors : pt is the wrong size\n"); 
  Vector<double> distances(N);
  Vector<double> Mi_pt(D);
  Vector<int> indices(N);
  for(int i=0;i<N;i++){
    Mi_pt = M.col(i)-pt;
    distances(i) = blas_NRM2(Mi_pt);
    indices(i) = i;
  } 
  argsort(distances,indices);
  nbrs = indices.SubVector(0,k);
}

bool compare_neighbors(const Vector<int>& nbrs1,
		       const Vector<int>& nbrs2){
  int k = nbrs1.size();
  if(k != nbrs2.size())
    return false;
  for(int i1=0;i1<k;i1++){
    int n1 = nbrs1(i1);
    bool found = false;
    for(int i2=0;i2<k;i2++){
      if(nbrs1(i1)==nbrs2(i2)){
	found = true;
	break;
      }
    }
    if(!found)
      return false;
  }
  return true;
}

void S_matrix(Matrix<double>& M){
  int N = M.ncols();
  int D = M.nrows();

  double r1,r2,x1,x2,x3;
  for(int i=0; i<N; i++){
    r1 = (std::rand() / (double(RAND_MAX) + 1) );
    r2 = (std::rand() / (double(RAND_MAX) + 1) );
    x1 = sin(r1);
    x2 = (r2/fabs(r1)) * (cos(r1)-1);
    x3 = r2*5;
    for(int j=0; j<D; j++){
      switch(i%3){
      case 0:
	M(j,i) = x1;
      case 1:
	M(j,i) = x2;
      case 2:
	M(j,i) = x3; 
      }
      if(j>3)
	M(j,i) += sin( 3*M(j-1,i) );
    }
  }
}

void rand_matrix(Matrix<double>& M){
  int D = M.nrows();
  int N = M.ncols();
  for(int i=0;i<D;i++){
    for(int j=0;j<N;j++){ 
      M(i,j) = (std::rand() / (double(RAND_MAX) + 1) );
    }
  } 
}
  


int main(){
  int D = 100;
  int N = 2000;
  int k = 30;
  Matrix<double> M(D,N);
  
  int rseed = time(NULL);
  std::cout << "rseed = " << rseed << "\n";
  std::cout << "----------------------------------------\n";
  srand ( rseed );
  std::cout << k << " neighbors of " 
	    << N << " points in " 
	    << D << " dimensions\n";
  
  //rand_matrix(M);
  S_matrix(M);
  //std::cout << M; 

  Vector<int> nbrs1(k);
  Vector<int> nbrs2(k);
  Vector<double> pt(D);
  
  clock_t start,end;

  start = clock();
  //BallTree BT(M);
  BallTree<double> BT(M);
  end = clock();
  std::cout << "   BallTree construction: "  //<< std::scientific
	    << (end-start)*1.0/CLOCKS_PER_SEC << " sec\n";
		  
  start = clock();
  for(int i=0;i<N;i++){
    pt = M.col(i);
    BT.knn_search(pt,nbrs1);
  }
  end = clock();
  std::cout << "   BallTree neighbors:    " //<< std::scientific
	    << (end-start)*1.0/CLOCKS_PER_SEC << " sec\n";
  
  start = clock(); 
  for(int i=0;i<N;i++){
    pt = M.col(i);
    compute_neighbors(M,pt,nbrs2);
  }
  end = clock();
  std::cout << "   brute-force neighbors: " //<< std::scientific
	    << (end-start)*1.0/CLOCKS_PER_SEC << " sec\n";
  
  
  std::cout << "rseed = " << rseed << "\n";
  if( compare_neighbors(nbrs1,nbrs2) ){
    std::cout << "neighbors are the same!\n";
  }else{
    std::cout << "neighbors are different!\n";
    std::cout << "ball: " << nbrs1;
    std::cout << "brute " << nbrs2;
  }
}
