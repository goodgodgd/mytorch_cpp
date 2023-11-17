#pragma once

#include "mytorch/tensor.hpp"

class Optimizer
{
  float lr_;

public:
  Optimizer(const float lr) : lr_(lr) {}
  void optimize(SpTensor y)
  {
    // TODO
  }
};
