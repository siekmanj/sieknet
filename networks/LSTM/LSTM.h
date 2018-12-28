#ifndef LSTM_H
#define LSTM_H

#include <stdlib.h>
#include <stdio.h>

#ifndef UNROLL_LENGTH
#define UNROLL_LENGTH 25
#endif

#define createLSTM(...) lstm_from_arr((size_t[]){__VA_ARGS__}, sizeof((size_t[]){__VA_ARGS__})/sizeof(size_t))

typedef struct gate{
	float *output;
	float *dOutput;
	float *gradient;
	float *weights;
	float bias;
	float *weight_updates;
	float bias_update;

} Gate;

typedef struct cell{
	Gate input_nonl;
	Gate input_gate;
	Gate forget_gate;
	Gate output_gate;
	float *output;
	float *dOutput;
	float *state;
	float *dstate;
	float *gradient;
} Cell;

typedef struct lstm_layer{
	Cell *cells;
	float *hidden;
	float *output;
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
//	float *input;
//	float *output;
	float **cost_gradients;
	LSTM_layer *head;
	LSTM_layer *tail;
	size_t t;
	int collapse;
	double plasticity;
	
} LSTM;

LSTM lstm_from_arr(size_t *arr, size_t len);
//LSTM loadLSTMFromFile(const char *filename);

//float step(LSTM_layer *, float *, float *);
void forward(LSTM *, float *);
void backward(LSTM *);
float quadratic_cost(LSTM *, float *);
//float step(LSTM *n, int label;

#endif
