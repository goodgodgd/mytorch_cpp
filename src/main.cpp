#include "mytorch/module.hpp"
#include "mytorch/dataset.hpp"
#include "mytorch/trainer.hpp"

int main()
{
  Dataset dataset(3);
  auto model = std::make_shared<Model>(std::vector<int>({ 3, 5, 4 }), true);
  auto loss = std::make_shared<DotLoss>(4);
  Trainer trainer(model, loss);
  trainer.train(dataset, 5);

  return 0;
}
