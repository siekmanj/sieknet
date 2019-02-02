#ifndef RNN_H
#define RNN_H

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "MLP.h"

// some magic to allow arbitrary numbers of parameters
#define create_rnn(...) rnn_from_arr((size_t[]){__VA_ARGS__}, sizeof((size_t[]){__VA_ARGS__})/sizeof(size_t))

typedef struct rnn_layer{
	Neuron *neurons;
	float *z;
	float *lout;
	float **gradient
	float **output;
	float **input;
	size_t size;
	size_t input_dimension;
	void (*logistic)(const float *, float *, size_t);

} RNN_layer;

typedef struct rnn{
	RNN_layer *layers;
	size_t depth;
	size_t num_params;
	size_t input_dimension;
	size_t output_dimension;
	size_t guess;

	float learning_rate;
	float *params;
	float *output;

	float **cost_gradient;
	float (*cost_fn)(float *y, const float *l, float *dest, size_t);
} RNN;

RNN rnn_from_arr(size_t arr[], size_t size);
RNN load_rnn(const char *filename);
void save_rnn(RNN *, const char *filename);

void rnn_forward(RNN *, float *);
float rnn_cost(RNN *, float *);
void rnn_backward(RNN *);

#endif
