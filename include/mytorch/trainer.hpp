#pragma once

#include "mytorch/module.hpp"
#include "mytorch/dataset.hpp"

class Trainer
{
  SpModule model_;
  SpModule loss_;

public:
  Trainer(SpModule model, SpModule loss) : model_(model), loss_(loss) {}
  void train(const Dataset& dataset, const int steps)
  {
    for (int i = 0; i < steps; ++i)
    {
      auto x = dataset.next();
      auto z = model_->forward(x);
      auto l = loss_->forward(z);
      l->backward();
      cout << "STEP: " << i << endl;
    }
  }
};
