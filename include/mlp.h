#ifndef MLP_H
#define MLP_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

// some magic
//#define MAX_BATCH_SIZE 500
#define create_mlp(...) mlp_from_arr((size_t[]){__VA_ARGS__}, sizeof((size_t[]){__VA_ARGS__})/sizeof(size_t))

typedef struct neuron{
	//float input; //needed for efficient softmax
	float *weight_grads;
	float *bias_grad;
	float *weights;
	float *bias;
} Neuron;

typedef struct layer{
	Neuron *neurons;
	float *gradient;
	float *z;
	//float **input;
	//float **output;
	float *output;
	float *input;
	size_t size;
	size_t input_dimension;
	void (*logistic)(const float *, float *, size_t);
} MLP_layer;


typedef struct mlp{
	MLP_layer *layers;
	size_t depth;
	//size_t b;
	//size_t batch_size;
	size_t num_params;
	size_t input_dimension;
	size_t output_dimension;
	size_t guess;

	float learning_rate;
	float *params;
	float *param_grad;
	float *output;

	float *cost_gradient;
	float (*cost_fn)(float *y, const float *l, float *dest, size_t);
} MLP;

MLP mlp_from_arr(size_t arr[], size_t size);
MLP load_mlp(const char *filename);

MLP_layer create_MLP_layer(size_t, size_t, float *, float *, void(*)(const float *, float *, size_t));

void mlp_forward(MLP *, float *);
void mlp_backward(MLP *);
float mlp_cost(MLP *, float *);

void save_mlp(const MLP *n, const char* filename);

void xavier_init(float *, size_t, size_t);
void zero_2d_arr(float **, size_t, size_t);

//These are activation functions
void hypertan(const float *, float *, const size_t);
void sigmoid(const float *, float *, const size_t);
void softmax(const float *, float *, const size_t);
void relu(const float *, float *, const size_t);

float inner_product(const float *, const float *, size_t);

float cross_entropy_cost(float *, const float *, float *, size_t);
void dealloc_mlp(MLP *);

#endif
