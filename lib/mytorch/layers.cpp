
/*
#include "mytorch/layers.hpp"

template <uint8_t D>
SpTensor<D> Linear::forwardImpl(SpTensor<D> x)
{
  SpTensor y(new Tensor);
  y->tnsr = w_->tnsr * x->tnsr + b_->tnsr;
  return y;
}

template <uint8_t D>
void Linear::gradient(SpTensor<D> x, SpTensor<D> y)
{
  // x.grad = dy/dx
  x->grad = w_->tnsr;
  // w_.grad = dy/dw
  w_->grad =
}

template <uint8_t D>
SpTensor<D> L2Loss::forwardImpl(SpTensor<D> x)
{
  SpTensor y(new Tensor);
  y->tnsr = x->tnsr.transpose() * x->tnsr.transpose();
  return y;
}

template <uint8_t D>
void L2Loss::gradient(SpTensor<D> x, SpTensor<D> y)
{
  // x.grad = dy/dx
  // w_.grad = dy/dw
}
*/
