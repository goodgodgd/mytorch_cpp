#pragma once

#include "mytorch/module.hpp"
#include "mytorch/dataset.hpp"
#include "mytorch/optimizer.hpp"

class Trainer
{
  UpModule model_;
  UpModule loss_;
  std::unique_ptr<Optimizer> optimizer_;

public:
  Trainer()
    : model_(new Model(std::vector<int>({ 3, 5, 4 }), true))
    , loss_(new DotLoss(4))
    , optimizer_(new Optimizer(0.001f))
  {
  }

  void train(const Dataset& dataset, const int steps)
  {
    for (int i = 0; i < steps; ++i)
    {
      cout << "===== STEP: " << i << endl;
      auto x = dataset.next();
      auto z = model_->forward(x);
      auto l = loss_->forward(z);
      cout << "loss x " << l->parm << endl;
      l->backward();
      optimizer_->optimize(l);
    }
  }
};
