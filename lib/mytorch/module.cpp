#include "mytorch/module.hpp"

SpTensor Linear::forwardImpl(SpTensor x)
{
  SpTensor y(new Tensor);
  y->parm = xt::linalg::tensordot(w_->parm, x->parm, 1) + b_->parm;
  cout << "Linear::forward" << y->parm << endl;
  return y;
}

void Linear::gradient(SpTensor x, SpTensor y)
{
  // x (N), y (M)
  // x.grad = dy/dx = w (M x N)
  x->grad = w_->parm;
  // w_.grad = dy/dw (M x M x N)
  int M = y->parm.shape()[0];
  int N = x->parm.shape()[0];
  w_->grad = xt::zeros<DType>({ M, M, N });
  for (int i = 0; i < M; ++i)
    xt::view(w_->grad, i, i, xt::all()) = x->parm;
}

SpTensor DotLoss::forwardImpl(SpTensor x)
{
  SpTensor y(new Tensor);
  y->parm = xt::pow(xt::linalg::dot(d_->parm, x->parm), 2.f);
  return y;
}

void DotLoss::gradient(SpTensor x, SpTensor y)
{
  // x (N), y (1)
  // x.grad = dy/dx = 2*d*d.T*x (N)
  x->grad = 2 * xt::linalg::dot(d_->parm, x->parm) * d_->parm;
}

SpTensor Model::forwardImpl(SpTensor x)
{
  SpTensor y = x;
  for (auto& layer : layers)
    y = layer->forward(y);
  return y;
}

void Model::gradient(SpTensor x, SpTensor y)
{
  // gradients of layers are computed in layer->forward() in forwardImpl()
}
