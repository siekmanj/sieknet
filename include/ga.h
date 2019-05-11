#ifndef GA_H
#define GA_H

#include <logistic.h>
#include <stdio.h>
#include <conf.h>

typedef enum mutation_t {NONE, BASELINE, SAFE, MOMENTUM, SAFE_MOMENTUM, AGGRESSIVE} Mutation_type;
typedef enum network_t {mlp, rnn, lstm} Network_type;

typedef struct member_{
	float performance;
  float *params;
  float *param_grad;
	float *momentum;
  size_t num_params;
} Member;

typedef struct pool_{
  Network_type network_type;
  Mutation_type mutation_type;

	float mutation_rate;
	float noise_std;
  float elite_percentile;

	Member **members;

  int crossover;
  size_t pool_size;
} Pool;

Pool create_pool(float *, size_t, size_t);

void evolve_pool(Pool *p);

void sensitivity_gradient(float *, const float *, Nonlinearity, size_t);

void sort_pool(Pool *p);
void cull_pool(Pool *p);
void breed_pool(Pool *p);

#endif
