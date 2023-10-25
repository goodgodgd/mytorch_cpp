#include <eigen3/Eigen/Eigen>
#include <vector>
#include <memory>
#include <iostream>
using std::cout;
using std::endl;

struct Tensor
{
  Eigen::MatrixXf tnsr;
  Eigen::MatrixXf grad;
  std::vector<std::shared_ptr<Tensor>> back_links;

  void Backward()
  {
    for (auto t : back_links)
    {
      t->grad = this->grad * t->grad;
      t->Backward();
      cout << "backward: (" << this->grad.rows() << ", " << this->grad.cols()
           << ") x (" << t->grad.rows() << ", " << t->grad.cols() << ")"
           << endl;
    }
  }
};

using SpTensor = std::shared_ptr<Tensor>;

class LayerBase
{
protected:
  bool train_mode_;

public:
  LayerBase() : train_mode_(false) {}
  void SetTrainMode(bool train_mode) { train_mode_ = train_mode; }
  virtual SpTensor Forward(SpTensor x) = 0;
  virtual void Gradient(SpTensor x, SpTensor y) = 0;
};

class TrainableLayer : public LayerBase
{
protected:
  SpTensor w_;
  SpTensor b_;

public:
  TrainableLayer() : LayerBase(), w_(new Tensor), b_(new Tensor) {}
  virtual void InitParams() = 0;
};

class Linear : public TrainableLayer
{
public:
  Linear(int in_feat, int out_feat) : TrainableLayer()
  {
    w_->tnsr = Eigen::MatrixXf::Random(out_feat, in_feat) * 0.1f;
    b_->tnsr = Eigen::VectorXf::Zero(out_feat);
  }

  virtual void InitParams() override {}

  virtual SpTensor Forward(SpTensor x) override;
  virtual void Gradient(SpTensor x, SpTensor y) override;
};
