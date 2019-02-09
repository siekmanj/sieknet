#ifndef LSTM_H
#define LSTM_H

#include <stdlib.h>
#include <stdio.h>
#include "mlp.h"

#ifndef MAX_UNROLL_LENGTH
#define MAX_UNROLL_LENGTH 800
#endif

#define create_lstm(...) lstm_from_arr((size_t[]){__VA_ARGS__}, sizeof((size_t[]){__VA_ARGS__})/sizeof(size_t))

typedef struct gate{
	float *output;
	float *dOutput;
	float *gradient;
	float *weights;
	float *bias;

	float *weight_grad;
	float *bias_grad;
} Gate;

typedef struct cell{
	Gate input_nonl;
	Gate input_gate;
	Gate forget_gate;
	Gate output_gate;
	float loutput;
	float lstate;
	float *state;
	float *dstate;
	float *dOutput;
} Cell;

typedef struct lstm_layer{
	Cell *cells;
	float **output;
	float **input;
	float **input_gradient;
	size_t input_dimension;
	size_t size;
} LSTM_layer;

typedef struct lstm{
	float *params;
	float *param_grad;
	float **network_gradient;
	float **network_input;
	
	//int collapse;
	int stateful;
	int guess;
	
	size_t input_dimension;
	size_t output_dimension;

	size_t num_params;
	size_t seq_len;
	size_t depth;
	size_t t;

	//float learning_rate;
	LSTM_layer *layers;

	MLP output_layer;
	void (*output_logistic)(const float *, float *, size_t);
	float (*cost_fn)(float *y, const float *l, float *dest, size_t);
	
} LSTM;

LSTM lstm_from_arr(size_t *arr, size_t len);
LSTM load_lstm(const char *filename);
void save_lstm(LSTM *n, const char *filename);

void lstm_forward(LSTM *, float *);
void lstm_backward(LSTM *);
float lstm_cost(LSTM *, float *);

void wipe(LSTM *);

void dealloc_lstm(LSTM *);

#endif
