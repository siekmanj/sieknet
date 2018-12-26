#ifndef LSTM_H
#define LSTM_H

#include <stdlib.h>
#include <stdio.h>

#ifndef UNROLL_LENGTH
#define UNROLL_LENGTH 50
#endif

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
	float **inputs;
	float **input_gradients;
	size_t input_dimension;
	size_t size;
	size_t t;
	double plasticity;
} LSTM_layer;


LSTM_layer createLSTM_layer(size_t input_dimension, size_t cells);
//LSTM loadLSTMFromFile(const char *filename);

float step(LSTM_layer *, float *, float *);
void process_cell(Cell *c);
//float step(LSTM *n, int label;

#endif
