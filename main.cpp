#include "include/mytorch/layers.hpp"

int main()
{
  SpTensor x(new Tensor);
  x->tnsr = Eigen::VectorXf(3);
  x->tnsr << 1, 2, 3;
  Linear layer(3, 3);
  SpTensor y = layer.Forward(x);
  SpTensor loss(new Tensor);
  loss->tnsr = y->tnsr.transpose() * y->tnsr;

  return 0;
}
