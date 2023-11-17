#pragma once

#include <xtensor/xrandom.hpp>

#include "mytorch/tensor.hpp"

class Module
{
protected:
  bool train_mode_;
  virtual SpTensor forwardImpl(SpTensor x) = 0;
  virtual void gradient(SpTensor x, SpTensor y) = 0;

public:
  Module(bool train_mode = false) : train_mode_(train_mode) {}
  virtual ~Module() {}
  void setTrainMode(bool train_mode) { train_mode_ = train_mode; }
  virtual SpTensor forward(SpTensor x)
  {
    SpTensor y = forwardImpl(x);
    if (train_mode_)
      gradient(x, y);
    return y;
  }
};

using UpModule = std::unique_ptr<Module>;

class Linear : public Module
{
  SpTensor w_;
  SpTensor b_;

public:
  Linear(uint32_t in_dim, uint32_t out_dim, bool train_mode)
    : Module(train_mode)
    , w_(new Tensor("lin_w", train_mode))
    , b_(new Tensor("lin_b", train_mode))
  {
    w_->parm = xt::random::randn<DType>({ out_dim, in_dim });
    b_->parm = xt::zeros<DType>({ out_dim });
  }
  virtual ~Linear() {}
  virtual SpTensor forwardImpl(SpTensor x) override;
  virtual void gradient(SpTensor x, SpTensor y) override;
};

class Model : public Module
{
  std::vector<UpModule> layers;

public:
  Model(std::vector<int> dims, bool train_mode) : Module(train_mode)
  {
    layers.push_back(std::make_unique<Linear>(dims[0], dims[1], train_mode));
    layers.push_back(std::make_unique<Linear>(dims[1], dims[2], train_mode));
  }
  virtual ~Model() {}
  virtual SpTensor forwardImpl(SpTensor x) override;
  virtual void gradient(SpTensor x, SpTensor y) override;
};

class DotLoss : public Module
{
  SpTensor d_;

public:
  DotLoss(uint32_t in_dim) : Module(true), d_(new Tensor("dot", false))
  {
    d_->parm = xt::random::rand<DType>({ in_dim });
    cout << "[DotLoss] d=" << d_->parm << endl;
  }
  virtual ~DotLoss() {}
  virtual SpTensor forwardImpl(SpTensor x) override;
  virtual void gradient(SpTensor x, SpTensor y) override;
};
