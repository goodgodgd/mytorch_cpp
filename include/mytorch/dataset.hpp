#pragma once

#include "mytorch/tensor.hpp"

class Dataset
{
  SpTensor x;

public:
  Dataset(uint32_t dim) : x(new Tensor)
  {
    x->parm = xt::random::randint({ dim }, 0, 5);
  }
  SpTensor next() const { return x; }
};
