#include <eigen3/Eigen/Eigen>
#include <vector>
#include <memory>
#include <iostream>
#include <cassert>
using std::cout;
using std::endl;
using TensorDtype = float;
using TensorIndex = uint32_t;

class Tensor
{
protected:
  Eigen::MatrixXf data_;
  Eigen::RowVectorXi shape_;

public:
  Tensor() {}
  Tensor(const Tensor &o) : data_(o.data()), shape_(o.shape()) {}
  Tensor(const Eigen::MatrixXf &data_in, Eigen::RowVectorXi shape_in) : data_(data_in), shape_(shape_in) {}
  Tensor(const Eigen::RowVectorXi shape_in) : shape_(shape_in)
  {
    cout << "shape_in" << shape_in << endl;
    if (rank() == 1)
      data_ = Eigen::MatrixXf::Zero(shape(0), 1);
    else if (rank() == 2)
      data_ = Eigen::MatrixXf::Zero(shape(0), shape(1));
    else
      data_ = Eigen::MatrixXf::Zero(size(), 1);
    int a = 1;
  }

  Eigen::RowVectorXi shape() const { return shape_; }
  TensorIndex shape(TensorIndex i) const { return shape_(i); }
  const Eigen::MatrixXf data() const { return data_; }
  TensorIndex rank() const { return shape_.size(); }
  TensorIndex size() const
  {
    TensorIndex sz = 1;
    for (auto dim : shape_)
      sz *= dim;
    return sz;
  }
  TensorDtype &operator()(TensorIndex i0)
  {
    assert(rank() == 1);
    return data_(i0, 0);
  }
  const TensorDtype &operator()(TensorIndex i0) const
  {
    assert(rank() == 1);
    return data_(i0, 0);
  }
  TensorDtype &operator()(TensorIndex i0, TensorIndex i1)
  {
    assert(rank() == 2);
    return data_(i0, i1);
  }
  const TensorDtype &operator()(TensorIndex i0, TensorIndex i1) const
  {
    assert(rank() == 2);
    return data_(i0, i1);
  }
  TensorDtype &operator()(TensorIndex i0, TensorIndex i1, TensorIndex i2)
  {
    assert(rank() == 3);
    TensorIndex index = i0 * shape(0) * shape(1) + i1 * shape(1) + i2;
    assert(index < size());
    return data_(index, 0);
  }
  const TensorDtype &operator()(TensorIndex i0, TensorIndex i1, TensorIndex i2) const
  {
    assert(rank() == 3);
    TensorIndex index = i0 * shape(0) * shape(1) + i1 * shape(1) + i2;
    assert(index < size());
    return data_(index, 0);
  }

  Tensor operator+(const Tensor &o) const { return Tensor(this->data() + o.data(), this->shape()); }
  Tensor operator-(const Tensor &o) const { return Tensor(this->data() - o.data(), this->shape()); }
  Tensor operator*(const Tensor &o) const { return *this; }
};

class GradTensor : public Tensor
{
protected:
  Tensor grad_;
  std::vector<std::shared_ptr<GradTensor>> back_links_;

public:
  GradTensor() : Tensor() {}
  GradTensor(const Eigen::RowVectorXi shape) : Tensor(shape) {}
  GradTensor(const GradTensor &o) : Tensor(o), grad_(o.grad()), back_links_(o.back_links()) {}

  Tensor &grad() { return grad_; }
  const Tensor grad() const { return grad_; }
  const std::vector<std::shared_ptr<GradTensor>> back_links() const { return back_links_; }

  void backward()
  {
    for (auto back : back_links_)
    {
      back->grad() = this->grad() * back->grad();
      back->backward();
      cout << "backward: (" << this->grad().shape(0) << ", " << this->grad().shape(1)
           << ") x (" << back->grad().shape(0) << ", " << back->grad().shape(1) << ")"
           << endl;
    }
  }
};

void testTensor()
{
  Tensor a({2, 3});
  a(0, 1) = 1;
  Tensor b({2, 3});
  b(1, 0) = 2;
  Tensor c = a + b;
  cout << c.data() << endl;
}
