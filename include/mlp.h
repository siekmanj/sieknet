#ifndef MLP_H
#define MLP_H


#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <conf.h>
#include <nonlinear.h>

#ifdef GPU
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#endif

#define create_mlp(...) mlp_from_arr((size_t[]){__VA_ARGS__}, sizeof((size_t[]){__VA_ARGS__})/sizeof(size_t), 1)

#ifndef GPU
typedef struct neuron{
	float *weight_grad;
	float *bias_grad;
	float *weights;
	float *bias;
} Neuron;
#endif

typedef struct layer{
#ifndef GPU
	Neuron *neurons;
	float *gradient;
	float *z;
	float *output;
	float *input;
#else
	int param_offset;
  cl_mem gradient;
  cl_mem z;
  cl_mem output;
  cl_mem input;
	cl_program prog;
#endif
	Nonlinearity logistic;
	size_t size;
	size_t input_dimension;
} MLP_layer;


typedef struct mlp{
	MLP_layer *layers;
	size_t depth;
	size_t num_params;
	size_t input_dimension;
	size_t output_dimension;
	size_t guess;

	float learning_rate;
	float *params;
	float *output;
#ifndef GPU
	float *param_grad;
#else
	cl_mem gpu_params;
	cl_mem param_grad;
	cl_mem network_input;
	cl_mem network_grad;
#endif
	float *cost_gradient;
	float (*cost_fn)(float *y, const float *l, float *dest, size_t);
} MLP;

MLP mlp_from_arr(size_t arr[], size_t size, int initialize);
MLP load_mlp(const char *filename);

#ifndef GPU
MLP_layer cpu_create_MLP_layer(size_t, size_t, float *, float *, Nonlinearity);
void cpu_mlp_layer_forward(MLP_layer *, float *);
void cpu_mlp_layer_backward(MLP_layer *, float *);
#else
MLP_layer gpu_create_MLP_layer(size_t, size_t, float *, int, Nonlinearity);
void gpu_mlp_layer_forward(MLP_layer *, cl_mem, cl_mem);
#endif

void mlp_forward(MLP *, float *);
void mlp_backward(MLP *);
float mlp_cost(MLP *, float *);

void save_mlp(MLP *n, const char* filename);
void dealloc_mlp(MLP *);

void xavier_init(float *, size_t, size_t);
void zero_2d_arr(float **, size_t, size_t);

//These are activation functions
#ifdef GPU
cl_kernel logistic_kernel, softmax_sum_kernel, softmax_kernel;
void mlp_kernel_setup();
#endif
float inner_product(const float *, const float *, size_t);
float cross_entropy_cost(float *, const float *, float *, size_t);
#endif
