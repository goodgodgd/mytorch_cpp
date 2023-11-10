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
      t->grad = xt::linalg::tensordot(this->grad, t->grad, 1);
      t->backward();
    }
  }
};

uint32_t Tensor::count = 0;

using SpTensor = std::shared_ptr<Tensor>;
