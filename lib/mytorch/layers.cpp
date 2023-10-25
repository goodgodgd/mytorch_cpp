#include "mytorch/layers.hpp"

SpTensor Linear::Forward(SpTensor x)
{
  SpTensor y;
  y->tnsr = w_->tnsr * x->tnsr + b_->tnsr;
  if (train_mode_)
    Gradient(x, y);
  y->back_links.push_back(x);
  y->back_links.push_back(w_);
  return y;
}

void Linear::Gradient(SpTensor x, SpTensor y)
{
  // x.grad = dy/dx
  // w_.grad = dy/dw
}
