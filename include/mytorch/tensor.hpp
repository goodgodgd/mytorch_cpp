#pragma once

#include <vector>
#include <memory>
#include <string>
#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>

using std::cout;
using std::endl;
using DType = float;

std::string shapeToString(const xt::xarray<DType>& x)
{
  auto shape = x.shape();
  std::string str = "(";
  for (auto e : shape)
    str += std::to_string(e) + ",";
  str += ')';
  return str;
}

struct Tensor
{
  std::string name;
  bool trainable;
  xt::xarray<DType> parm;
  xt::xarray<DType> grad;
  std::vector<std::shared_ptr<Tensor>> back_links;
  static uint32_t count;

  Tensor(std::string _name = "", bool _train = false)
    : name(_name), trainable(_train)
  {
    if (name.empty())
      name = "tnsr";
    name += "_" + std::to_string(count++);
  }

  void backward()
  {
    for (auto t : back_links)
    {
      // cout << "[backward] " << name << endl
      //      << shapeToString(this->grad) << shapeToString(t->grad) << endl;
      t->grad = xt::linalg::tensordot(this->grad, t->grad, 1);
      t->backward();
    }
  }
};

uint32_t Tensor::count = 0;

using SpTensor = std::shared_ptr<Tensor>;

std::string tensorShapeToString(SpTensor x)
{
  std::string str = "[" + x->name + ":p";
  str += shapeToString(x->parm) + ", g";
  str += shapeToString(x->grad) + "]";
  return str;
}
