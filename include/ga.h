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

typedef struct ga_{
  Network_type network_type;
  Mutation_type mutation_type;

	float mutation_rate;
	float noise_std;
  float elite_percentile;

	Member **members;

  int crossover;
  size_t size;
} GA;

GA create_ga(float *, size_t, size_t);

void ga_evolve(GA *);

void sensitivity_gradient(float *, const float *, Nonlinearity, size_t);

#endif
