#ifndef LSTM_H
#define LSTM_H

#include <stdlib.h>
#include <stdio.h>
#include "mlp.h"

#ifndef MAX_UNROLL_LENGTH
#define MAX_UNROLL_LENGTH 200
#endif

#define create_lstm(...) lstm_from_arr((size_t[]){__VA_ARGS__}, sizeof((size_t[]){__VA_ARGS__})/sizeof(size_t))

typedef struct gate{
	float *output;
	float *dOutput;
	float *gradient;
	float *weights;
	float *bias;
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
	//float *output;
	float *dOutput;
	//float *gradient;
} Cell;

typedef struct lstm_layer{
	Cell *cells;
	//float *hidden;
	float **output;
	float **input;
	float **input_gradient;
	//struct lstm_layer *input_layer;
	//struct lstm_layer *output_layer;
	size_t input_dimension;
	size_t size;
	//size_t t;
	//double plasticity;
} LSTM_layer;

typedef struct lstm{
	float *params;
	float **network_gradient;
	float **network_input;
	
	int collapse;
	int stateful;
	int guess;
	
	size_t input_dimension;
	size_t output_dimension;

	size_t num_params;
	size_t seq_len;
	size_t depth;
	size_t t;

	float learning_rate;
	LSTM_layer *layers;

	MLP output_layer;
	float (*cost)(struct lstm *, float *);
	
} LSTM;

LSTM lstm_from_arr(size_t *arr, size_t len);
LSTM load_lstm(const char *filename);
void save_lstm(LSTM *n, const char *filename);

void lstm_forward(LSTM *, float *);
void lstm_backward(LSTM *);

void wipe(LSTM *);

#endif
