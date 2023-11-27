#include "mytorch/dataset.hpp"
#include "mytorch/trainer.hpp"

int main()
{
  Dataset dataset(3);
  Trainer trainer;
  trainer.train(dataset, 10);
  return 0;
}
