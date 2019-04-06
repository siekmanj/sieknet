#ifndef LSTM_H
#define LSTM_H

#include <stdlib.h>
#include <stdio.h>
#include <mlp.h>

#ifndef MAX_UNROLL_LENGTH
#define MAX_UNROLL_LENGTH 400
#endif

#define create_lstm(...) lstm_from_arr((size_t[]){__VA_ARGS__}, sizeof((size_t[]){__VA_ARGS__})/sizeof(size_t))

typedef struct lstm_layer{
#ifndef SIEKNET_USE_GPU

	float **input_nonl_z;
	float **input_gate_z;
	float **forget_gate_z;
	float **output_gate_z;

	float **input_nonl_output;
	float **input_gate_output;
	float **forget_gate_output;
	float **output_gate_output;

	float **input_nonl_gradient;
	float **input_gate_gradient;
	float **forget_gate_gradient;
	float **output_gate_gradient;

	float **input_gradient;
	float **cell_gradient;

	float **cell_state;
	float **cell_dstate;

	float **input;
	float **output;

	float *loutput;
	float *lstate;
#else
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

	cl_mem *input_gradient;
	cl_mem *cell_gradient;

	cl_mem *cell_state;
	cl_mem *cell_dstate;

	cl_mem *input;
	cl_mem *output;

	cl_mem loutput;
	cl_mem lstate;
#endif
	int param_offset;

	size_t input_dimension;
	size_t size;
} LSTM_layer;

typedef struct lstm{
#ifndef SIEKNET_USE_GPU
	float *params;
	float *param_grad;
	float **network_gradient;
	float **network_input;

	float *mlp_cost_gradient;
#else
	cl_mem params;
	cl_mem param_grad;
	cl_mem *network_gradient;
	cl_mem *network_input;

	cl_mem mlp_cost_gradient;
	cl_mem output_label;
#endif
	float *output;

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
	Costfn cost_fn;
} LSTM;

LSTM lstm_from_arr(size_t *arr, size_t len);
LSTM load_lstm(const char *filename);
void save_lstm(LSTM *n, const char *filename);

void lstm_forward(LSTM *, float *);
void lstm_backward(LSTM *);
float lstm_cost(LSTM *, float *);

void wipe(LSTM *);

void dealloc_lstm(LSTM *);

/*
 * In this file, lstm kernels are implemented as macros to be used in both the gpu and cpu
 * implementation of Sieknet.
 *
 * This was a design decision to emphasize re-use of code, and enforce under-the-hood
 * homogeneity across CPU/GPU implementations. Unfortunately, OpenCL does not allow address
 * space changes (i.e., passing a __global pointer to a function that takes a pointer), which
 * necessitated the use of macros to provide an implementation that could be reused on the 
 * GPU as well as the CPU. If this were not the case, the below code would have been
 * implemented as a series of functions.
 */

/*<<KERNEL START>>*/

#define agnostic_lstm_forward_kernel(input_nonl, input_gate, forget_gate, output_gate, cell_state, cell_lstate, layer_output, i) \
	if(cell_lstate[i] > SIEKNET_MAX_STATE)                                                \
		cell_state[i] = input_nonl[i] * input_gate[i] + forget_gate[i] * SIEKNET_MAX_STATE; \
	else if(cell_lstate[i] < -SIEKNET_MAX_STATE)                                          \
		cell_state[i] = input_nonl[i] * input_gate[i] - forget_gate[i] * SIEKNET_MAX_STATE; \
	else                                                                                  \
		cell_state[i] = input_nonl[i] * input_gate[i] + forget_gate[i] * cell_lstate[i];    \
	layer_output[i] = HYPERTAN(cell_state[i]) * output_gate[i];                           \
	no_op()



#define agnostic_lstm_input_gradient_kernel(input_nonl_grad, input_gate_grad, forget_gate_grad, output_gate_grad, params, input_gradient, size, input_dimension, layer_param_offset, skipdist, i) \
	input_gradient[i] = 0.0f;                                        \
	for(int j = 0; j < size; j++){                                   \
		const int params_per_gate = input_dimension+1;                 \
		const int w_idx = layer_param_offset + (skipdist * j) + i;     \
		input_gradient[i] += input_nonl_grad[j]  * params[w_idx + 0 * params_per_gate + 1]; \
		input_gradient[i] += input_gate_grad[j]  * params[w_idx + 1 * params_per_gate + 1]; \
		input_gradient[i] += forget_gate_grad[j] * params[w_idx + 2 * params_per_gate + 1]; \
		input_gradient[i] += output_gate_grad[j] * params[w_idx + 3 * params_per_gate + 1]; \
	} \
  no_op()

/*<<KERNEL END>>*/
#endif
