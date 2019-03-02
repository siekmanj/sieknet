#ifndef MLP_H
#define MLP_H


#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <conf.h>

#ifdef GPU
#include <CL/cl.h>
#endif

#define create_mlp(...) mlp_from_arr((size_t[]){__VA_ARGS__}, sizeof((size_t[]){__VA_ARGS__})/sizeof(size_t))

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
	void (*logistic)(const float *, float *, size_t);
#else
  cl_mem gradient;
  cl_mem z;
  cl_mem output;
  cl_mem input;
  cl_kernel logistic;
#endif
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
#endif
	float *cost_gradient;
	float (*cost_fn)(float *y, const float *l, float *dest, size_t);
} MLP;

#ifdef GPU
void gpu_setup();
#endif

MLP mlp_from_arr(size_t arr[], size_t size);
MLP load_mlp(const char *filename);

MLP_layer cpu_create_MLP_layer(size_t, size_t, float *, float *, void(*)(const float *, float *, size_t));

void mlp_forward(MLP *, float *);
void mlp_backward(MLP *);
float mlp_cost(MLP *, float *);

void save_mlp(const MLP *n, const char* filename);

void xavier_init(float *, size_t, size_t);
void zero_2d_arr(float **, size_t, size_t);

//These are activation functions
#ifndef GPU
void hypertan(const float *, float *, const size_t);
void sigmoid(const float *, float *, const size_t);
void softmax(const float *, float *, const size_t);
void relu(const float *, float *, const size_t);

void dealloc_mlp(MLP *);
#else
cl_kernel linear, hypertan, sigmoid, relu;
#endif

float inner_product(const float *, const float *, size_t);

float cross_entropy_cost(float *, const float *, float *, size_t);

#endif
