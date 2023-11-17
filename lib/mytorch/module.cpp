#include "mytorch/module.hpp"

SpTensor Linear::forwardImpl(SpTensor x)
{
  SpTensor y(new Tensor("lin_y"));
  y->parm = xt::linalg::tensordot(w_->parm, x->parm, 1) + b_->parm;
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
  // b_.grad = dy/db = I
  b_->grad = xt::ones<DType>({ M, M });

  y->back_links.clear();
  y->back_links.push_back(x);
  y->back_links.push_back(w_);
  y->back_links.push_back(b_);
}

SpTensor DotLoss::forwardImpl(SpTensor x)
{
  SpTensor y(new Tensor("dot_y"));
  y->parm = xt::pow(xt::linalg::dot(d_->parm, x->parm), 2.f);
  return y;
}

void DotLoss::gradient(SpTensor x, SpTensor y)
{
  // x (N), y (1)
  // x.grad = dy/dx = 2*d*d.T*x (N)
  x->grad = 2 * xt::linalg::dot(d_->parm, x->parm) * d_->parm;
  int N = x->parm.shape()[0];
  x->grad = x->grad.reshape({ 1, N });
  // end output must have gradient=1
  y->grad = xt::xarray<DType>({ 1 });
  y->back_links.clear();
  y->back_links.push_back(x);
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
