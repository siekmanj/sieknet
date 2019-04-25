#ifndef GA_H
#define GA_H

#include <logistic.h>
#include <lstm.h>
#include <rnn.h>
#include <mlp.h>

typedef enum mutation_t { MUT_none, MUT_baseline, MUT_safe, MUT_momentum, MUT_safe_momentum } Mutation_type;
typedef enum network_t {mlp, rnn, lstm} Network_type;

typedef struct pool_{
  Network_type network_type;
  Mutation_type mutation_type;

	float mutation_rate;
	float step_size;
  float elite_percentile;

	void **members;

  size_t pool_size;
  size_t num_params;
  float *momentum;
} Pool;

LSTM *copy_lstm(LSTM *);
RNN *copy_rnn(RNN *);
MLP *copy_mlp(MLP *);

Pool create_pool(Network_type, void *, size_t);

void evolve(Pool *p);

void sort_pool(Pool *p);
void cull_pool(Pool *p);
void breed_pool(Pool *p);

#endif
