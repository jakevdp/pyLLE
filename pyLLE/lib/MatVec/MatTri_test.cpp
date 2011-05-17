#include "MatTri.h"

int main(){
  Matrix<double,TRI> M(4,3,9.0);
  
  std::cout << M << "\n";
  M.flip_unitdiag();
  std::cout << M.Transpose() << "\n";

  M.flip_unitdiag();
  M(1,1) = 3.0;
  M(1,2) = 4;
  std::cout << M.Transpose() << "\n";
  return 0;
}
