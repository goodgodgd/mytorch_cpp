#pragma once

#include "mytorch/tensor.hpp"

class Optimizer
{
  float lr_;

public:
  Optimizer(const float lr) : lr_(lr) {}
  void optimize(SpTensor y)
  {
    if (y->trainable)
    {
      y->parm -= lr_ * y->grad;
      cout << "optimize: " << y->name << ", parm" << y->parm << endl;
      //      << "grad" << y->grad << endl;
    }

    for (auto b : y->back_links)
      optimize(b);
  }
};
