#ifndef GA_H
#define GA_H

#include <logistic.h>
#include <lstm.h>
#include <rnn.h>
#include <mlp.h>

typedef enum mutation_t { MUT_none, MUT_baseline, MUT_safe, MUT_momentum, MUT_safe_momentum } Mutation_type;
typedef enum network_t {mlp, rnn, lstm} Network_type;

typedef struct mutation_ {
  Network_type network_type;
  Mutation_type mutation_type;

	float mutation_rate;
	float step_size;
  float elite_percentile;
  size_t pool_size;

  size_t num_params;
  float *momentum;

	void *(*recombine)(struct mutation_ m, void *a, void *b);
} Mutator;

typedef struct env_ {
  float *state;
  void *data;
  
  size_t action_space;
  size_t observation_space;

  void (*create)(struct env_ env);
  void (*dispose)(struct env_ env);
  void (*reset)(struct env_ env);
  void (*render)(struct env_ env);
  float (*step)(struct env_ env, float *action);
  
  
} Environment;


typedef LSTM** LSTM_pool;
typedef RNN**  RNN_pool;
typedef MLP**  MLP_pool;

LSTM *copy_lstm(LSTM *);
RNN *copy_rnn(RNN *);
MLP *copy_mlp(MLP *);

LSTM_pool create_lstm_pool(size_t, LSTM *, int);
RNN_pool create_rnn_pool(size_t, RNN *, int);
MLP_pool create_mlp_pool(size_t, MLP *, int);

void sort_lstm_pool(LSTM_pool p, size_t size);
void sort_rnn_pool(RNN_pool p, size_t size);
void sort_mlp_pool(MLP_pool p, size_t size);

void cull_lstm_pool(LSTM_pool p, Mutator m, size_t size);
void cull_rnn_pool(RNN_pool p, Mutator m, size_t size);
void cull_mlp_pool(MLP_pool p, Mutator m, size_t size);

void breed_lstm_pool(LSTM_pool p, Mutator m, size_t size);
void breed_rnn_pool(RNN_pool p, Mutator m, size_t size);
void breed_mlp_pool(MLP_pool p, Mutator m, size_t size);

Mutator create_mutator(Network_type, Mutation_type);

void set_sensitivity_gradient(float *, float *, Nonlinearity);

#endif


/*
 *  
 *
 *
 *
 */
