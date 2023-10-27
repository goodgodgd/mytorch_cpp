]#include "include/mytorch/tensor.hpp"

int main()
{
  Eigen::RowVectorXi shape = {2, 3};
  // Eigen::MatrixXf mat(shape(0), shape(1));
  cout << shape << endl;
  /*
  x->tnsr = Eigen::VectorXf(3);
  x->tnsr << 1, 2, 3;
  Linear layer(3, 3);
  SpTensor y = layer.forward(x);
  SpTensor loss(new Tensor);
  loss->tnsr = y->tnsr.transpose() * y->tnsr;
  */

  return 0;
}
