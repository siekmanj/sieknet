#ifndef GA_H
#define GA_H

#include <logistic.h>

typedef enum mutation_t { NONE, BASELINE, SAFE } MUTATION_TYPE;

typedef struct mutation {
	float *params;
	float mutation_rate;
	void (*recombine)(void *a, void *b);
} Mutator;

typedef struct safe_mutation {
	float *params;
	float *param_grad;
	float mutation_rate;
	void (*recombine)(void *a, void *b);
} Safe_Mutator;


//typedef LSTM** LSTM_pool;
//typedef RNN**  RNN_pool;
//typedef MLP**  MLP_pool;

LSTM copy_lstm(LSTM *);
RNN copy_rnn(RNN *);
MLP copy_mlp(MLP *);

LSTM lstm_recombinate(LSTM *, LSTM *, float, MUTATION_TYPE);
RNN  rnn_recombinate(RNN *, RNN *, float, MUTATION_TYPE);
MLP  mlp_recombinate(MLP *, MLP *, float, MUTATION_TYPE);

void set_sensitivity_gradient(float *, float *, Nonlinearity);

#endif


/*
 *  
 *
 *
 *
 */
