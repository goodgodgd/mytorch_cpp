#include "include/mytorch/layers.hpp"

int main()
{
  uint32_t x_dim = 5;
  uint32_t z_dim = 4;
  SpTensor x(new Tensor);
  x->parm = xt::random::randint({ x_dim }, 0, 3);
  Linear linear(x_dim, z_dim);
  linear.setTrainMode(true);
  DotLoss loss(4);
  loss.setTrainMode(true);
  for (int i = 0; i < 10; ++i)
  {
    auto z = linear.forward(x);
    auto l = loss.forward(z);
    l->backward();
    cout << "STEP: " << i << endl;
  }
  return 0;
}
