#ifndef MLP_H
#define MLP_H


#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <conf.h>
#include <logistic.h>
#include <mlp.kernel>

#ifdef SIEKNET_USE_GPU
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#endif

#define create_mlp(...) mlp_from_arr((size_t[]){__VA_ARGS__}, sizeof((size_t[]){__VA_ARGS__})/sizeof(size_t))

typedef struct mlp_layer{
#ifndef SIEKNET_USE_GPU
	//Neuron *neurons;
	float *input_gradient;
	float *z;
	float *output;
	float *input;
#else
  cl_mem input_gradient;
  cl_mem z;
  cl_mem output;
  cl_mem input;

	cl_program prog;
#endif
	int param_offset;
	size_t size;
	size_t input_dimension;
	Nonlinearity logistic;
} MLP_layer;


typedef struct mlp{
	MLP_layer *layers;
	size_t depth;
	size_t num_params;
	size_t input_dimension;
	size_t output_dimension;
	size_t guess;

	float learning_rate;
	float *output;

#ifndef SIEKNET_USE_GPU
	float *params;
	float *param_grad;

	float *cost_gradient;
#else
	cl_mem params;
	cl_mem param_grad;
	cl_mem network_input;
	cl_mem output_label;

	cl_mem cost_gradient;
#endif
	//float (*cost_fn)(float *y, const float *l, float *dest, size_t);
	Costfn cost_fn;
} MLP;

MLP mlp_from_arr(size_t arr[], size_t size);
MLP load_mlp(const char *filename);

#ifndef SIEKNET_USE_GPU
MLP_layer cpu_create_MLP_layer(const size_t, const size_t, float *, const int, const Nonlinearity);
void cpu_mlp_layer_forward(MLP_layer *, float *, float *);
void cpu_mlp_layer_backward(MLP_layer *, float *, float *, float *);
#else
MLP_layer gpu_create_MLP_layer(size_t, size_t, cl_mem, int, Nonlinearity);
void gpu_mlp_layer_forward(MLP_layer *, cl_mem, cl_mem);
void gpu_mlp_layer_backward(MLP_layer *, cl_mem, cl_mem, cl_mem);
#endif

void mlp_forward(MLP *, float *);
void mlp_backward(MLP *);
float mlp_cost(MLP *, float *);

void save_mlp(MLP *n, const char* filename);
void dealloc_mlp(MLP *);

void xavier_init(float *, size_t, size_t);
void zero_2d_arr(float **, size_t, size_t);

#ifdef SIEKNET_USE_GPU
float gpu_cost(cl_mem, cl_mem, cl_mem, size_t, Costfn);
//cl_kernel softmax_sum_kernel, softmax_kernel, zero_init_kernel;
void mlp_kernel_setup();
#endif
float cpu_cost(float *, float *, float *, size_t, Costfn);
float inner_product(const float *, const float *, size_t);
#endif
