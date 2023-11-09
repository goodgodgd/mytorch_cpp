#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor-blas/xlinalg.hpp"
#include <vector>
#include <memory>
#include <iostream>

using std::cout;
using std::endl;
using DType = float;

struct Tensor
{
  xt::xarray<DType> parm;
  xt::xarray<DType> grad;
  std::vector<std::shared_ptr<Tensor>> back_links;

  void backward()
  {
    for (auto t : back_links)
    {
      t->grad = xt::linalg::tensordot(this->grad, t->grad, 1);
      t->backward();
      // cout << "backward: " << this->grad.shape() << " x " << t->grad.shape()
      // << endl;
    }
  }
};

using SpTensor = std::shared_ptr<Tensor>;

class LayerBase
{
protected:
  bool train_mode_;
  virtual SpTensor forwardImpl(SpTensor x) = 0;
  virtual void gradient(SpTensor x, SpTensor y) = 0;

public:
  LayerBase() : train_mode_(false) {}
  void setTrainMode(bool train_mode) { train_mode_ = train_mode; }
  virtual SpTensor forward(SpTensor x)
  {
    SpTensor y = forwardImpl(x);
    if (train_mode_)
      gradient(x, y);
    return y;
  }
};

class Linear : public LayerBase
{
  SpTensor w_;
  SpTensor b_;

public:
  Linear(uint32_t in_dim, uint32_t out_dim)
    : LayerBase(), w_(new Tensor), b_(new Tensor)
  {
    w_->parm = xt::random::randn<DType>({ out_dim, in_dim }) * 0.1f;
    b_->parm = xt::zeros<DType>({ out_dim });
  }
  virtual SpTensor forwardImpl(SpTensor x) override;
  virtual void gradient(SpTensor x, SpTensor y) override;
};

class DotLoss : public LayerBase
{
  SpTensor d_;

public:
  DotLoss(uint32_t in_dim) : LayerBase(), d_(new Tensor)
  {
    d_->parm = xt::random::rand<DType>({ in_dim });
  }
  virtual SpTensor forwardImpl(SpTensor x) override;
  virtual void gradient(SpTensor x, SpTensor y) override;
};
