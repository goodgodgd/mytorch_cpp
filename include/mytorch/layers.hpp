#include <eigen3/Eigen/Eigen>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <vector>
#include <memory>
#include <iostream>

/*
using Eigen::Tensor;
using std::cout;
using std::endl;

template <uint8_t D>
struct GradTensor
{
  Tensor<float, D> tnsr;
  Tensor<float, D + 1> grad;
  std::vector<std::shared_ptr<GradTensor<D>>> back_links;

  void backward()
  {
    for (auto t : back_links)
    {
      t->grad = this->grad * t->grad;
      t->backward();
      cout << "backward: (" << this->grad.dimension(0) << ", " << this->grad.dimension(1)
           << ") x (" << t->grad.dimension(0) << ", " << t->grad.dimension(1) << ")"
           << endl;
    }
  }
};

template <uint8_t D>
using SpTensor = std::shared_ptr<GradTensor<D>>;
using SpTensor1 = SpTensor<1>;
using SpTensor2 = SpTensor<2>;
using SpTensor2 = SpTensor<2>;

// assuming D = input dim = output dim
template <uint8_t D>
class LayerBase
{
protected:
  bool train_mode_;
  virtual SpTensor<D> forwardImpl(SpTensor<D> x) = 0;

public:
  LayerBase() : train_mode_(false) {}
  void setTrainMode(bool train_mode) { train_mode_ = train_mode; }
  virtual SpTensor<D> forward(SpTensor<D> x)
  {
    SpTensor<D> y = forwardImpl(x);
    if (train_mode_)
      gradient(x, y);
    return y;
  }
  virtual void gradient(SpTensor<D> x, SpTensor<D> y) = 0;
};

template <uint8_t D>
class Linear : public LayerBase<2>
{
  SpTensor<D> w_;
  SpTensor<D> b_;

public:
  Linear(int in_feat, int out_feat) : LayerBase(), w_(new GradTensor<D>), b_(new GradTensor<D>)
  {
    w_->tnsr.setRandom();
    w_->tnsr *= 0.1f;
    b_->tnsr = Eigen::VectorXf::Zero(out_feat);
  }

  virtual SpTensor<D> forwardImpl(SpTensor<D> x) override;
  virtual void gradient(SpTensor<D> x, SpTensor<D> y) override;
};

template <uint8_t D>
class L2Loss : public LayerBase<D>
{
public:
  L2Loss() : LayerBase() {}
  virtual SpTensor<D> forwardImpl(SpTensor<D> x) override;
  virtual void gradient(SpTensor<D> x, SpTensor<D> y) override;
};
*/