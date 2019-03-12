#ifndef LSTM_H
#define LSTM_H

#include <stdlib.h>
#include <stdio.h>
#include "mlp.h"

#ifndef MAX_UNROLL_LENGTH
#define MAX_UNROLL_LENGTH 400
#endif

#define create_lstm(...) lstm_from_arr((size_t[]){__VA_ARGS__}, sizeof((size_t[]){__VA_ARGS__})/sizeof(size_t))

#ifndef GPU
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
#endif

typedef struct lstm_layer{
#ifndef GPU
	Cell *cells;
	float **output;
	float **input;
	float **input_gradient;
#else
	int param_offset;
	cl_mem *input_nonl_z;
	cl_mem *input_gate_z;
	cl_mem *forget_gate_z;
	cl_mem *output_gate_z;

	cl_mem *input_nonl_output;
	cl_mem *input_gate_output;
	cl_mem *forget_gate_output;
	cl_mem *output_gate_output;

	cl_mem *input_nonl_gradient;
	cl_mem *input_gate_gradient;
	cl_mem *forget_gate_gradient;
	cl_mem *output_gate_gradient;

	cl_mem *cell_state;

	cl_mem *input;
	cl_mem *output;

	cl_mem loutput;
	cl_mem lstate;
#endif

	size_t input_dimension;
	size_t size;
} LSTM_layer;

typedef struct lstm{
	float *output;
	float *params;
#ifndef GPU
	float *param_grad;
	float **network_gradient;
	float **network_input;
#else
	cl_mem gpu_params;
	cl_mem param_grad;
	cl_mem *network_gradient;
	cl_mem *network_input;
#endif

	int stateful;
	int guess;
	
	size_t input_dimension;
	size_t output_dimension;

	size_t num_params;
	size_t seq_len;
	size_t depth;
	size_t t;

	LSTM_layer *layers;

	MLP_layer output_layer;
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
