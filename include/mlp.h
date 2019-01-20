#ifndef MLP_H
#define MLP_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

// some magic
#define createMLP(...) mlp_from_arr((size_t[]){__VA_ARGS__}, sizeof((size_t[]){__VA_ARGS__})/sizeof(size_t))

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

void mlp_forward(MLP *, float *);
void mlp_backward(MLP *);

//void gradients_wrt_outputs(MLP_layer *);
//void mutate(MLP_layer *, float, float);

void save_mlp(const MLP *n, const char* filename);

float descend(MLP *n, float *, float *);
float backpropagate(MLP *n, float *);

//int bestGuess(MLP *n);

//These are activation functions
void hypertan(MLP_layer* layer); //Sometimes unstable
void sigmoid(MLP_layer* layer);
void softmax(MLP_layer* layer);

void dealloc_network(MLP *);

//void printOutputs(MLP_layer *layer);
//void prettyprint(MLP_layer *layer);
//void printActivationGradients(MLP_layer *layer);
//void printWeights(MLP_layer *layer);

#endif
