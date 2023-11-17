#pragma once

#include <ctime>
#include "mytorch/tensor.hpp"

class Dataset
{
  SpTensor x;

public:
  Dataset(uint32_t dim) : x(new Tensor)
  {
    xt::random::seed(time(NULL));
    x->parm = xt::random::randint({ dim }, 0, 5);
    x->parm = xt::random::randint({ dim }, 0, 5);
    cout << "[Dataset] d=" << x->parm << endl;
  }
  SpTensor next() const { return x; }
};
