#ifndef MLP_H
#define MLP_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

// some magic
#define create_mlp(...) mlp_from_arr((size_t[]){__VA_ARGS__}, sizeof((size_t[]){__VA_ARGS__})/sizeof(size_t))

typedef struct neuron{
	float input; //needed for efficient softmax
	float *weights;
	float *bias;
} Neuron;

typedef struct layer{
	//struct layer *input_layer;
	//struct layer *output_layer;
	Neuron *neurons;
	float *gradient;
	float *output;
	float *input;
	size_t size;
	size_t input_dimension;
	void (*logistic)(struct layer *);
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
	//size_t batch_size;
	float *cost_gradient;
	float (*cost)(struct mlp *, float *);
} MLP;

MLP mlp_from_arr(size_t arr[], size_t size);
MLP load_mlp(const char *filename);

MLP_layer create_MLP_layer(size_t, size_t, float *, void(*)(MLP_layer *));

void mlp_forward(MLP *, float *);
void mlp_backward(MLP *);

void mlp_layer_forward(MLP_layer *, float *);
void mlp_layer_backward(MLP_layer *, float *, float);

void save_mlp(const MLP *n, const char* filename);

float descend(MLP *n, float *, float *);
float backpropagate(MLP *n, float *);

void xavier_init(float *, size_t, size_t);


//These are activation functions
void hypertan(MLP_layer* layer);
void sigmoid(MLP_layer* layer);
void softmax(MLP_layer* layer);
void relu(MLP_layer* layer);

float inner_product(const float *, const float *, size_t);

float cross_entropy_cost(MLP *, float *);
void dealloc_mlp(MLP *);

#endif
