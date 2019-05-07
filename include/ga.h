#ifndef GA_H
#define GA_H

#include <logistic.h>
#include <lstm.h>
#include <rnn.h>
#include <mlp.h>


typedef enum mutation_t {NONE, BASELINE, SAFE, MOMENTUM, SAFE_MOMENTUM} Mutation_type;
typedef enum network_t {mlp, rnn, lstm} Network_type;

typedef struct member_{
	float performance;
	float *momentum;
	void *network;
} Member;

typedef struct pool_{
  Network_type network_type;
  Mutation_type mutation_type;
  int crossover;

	float mutation_rate;
	float noise_std;
  float elite_percentile;

	Member **members;

  size_t pool_size;
  size_t num_params;
  float *momentum;
} Pool;


LSTM *copy_lstm(LSTM *);
RNN *copy_rnn(RNN *);
MLP *copy_mlp(MLP *);

Pool create_pool(Network_type, void *, size_t);

void evolve_pool(Pool *p);

void sensitivity_gradient(float *, const float *, Nonlinearity, size_t);

void sort_pool(Pool *p);
void cull_pool(Pool *p);
void breed_pool(Pool *p);

#endif
