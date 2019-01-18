#ifndef LSTM_H
#define LSTM_H

#include <stdlib.h>
#include <stdio.h>
#include "MLP.h"

#ifndef MAX_UNROLL_LENGTH
#define MAX_UNROLL_LENGTH 1000
#endif

#define createLSTM(...) lstm_from_arr((size_t[]){__VA_ARGS__}, sizeof((size_t[]){__VA_ARGS__})/sizeof(size_t))

typedef struct gate{
	float *output;
	float *dOutput;
	float *gradient;
	float *weights;
	float bias;
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
	float *output;
	float *dOutput;
	float *gradient;
} Cell;

typedef struct lstm_layer{
	Cell *cells;
	float *output;
	float *hidden;
	float **inputs;
	float **input_gradients;
	struct lstm_layer *input_layer;
	struct lstm_layer *output_layer;
	size_t input_dimension;
	size_t size;
	size_t t;
	double plasticity;
} LSTM_layer;

typedef struct lstm{
	float **cost_gradients;
	double plasticity;
	size_t t;
	int collapse;
	int stateful;
	int seq_len;

	LSTM_layer *head;
	LSTM_layer *tail;

	MLP output_layer;
	
} LSTM;

LSTM lstm_from_arr(size_t *arr, size_t len);
LSTM loadLSTMFromFile(const char *filename);
void saveLSTMToFile(LSTM *n, char *filename);

void forward(LSTM *, float *);
float backward(LSTM *, float *);
float quadratic_cost(LSTM *, float *);
float cross_entropy_cost(LSTM *, float *);

void wipe(LSTM *);

#endif
