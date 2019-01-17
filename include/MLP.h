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
	struct layer *input_layer;
	struct layer *output_layer;
	Neuron *neurons;
	float *gradients;
	float *output;
	float *input;
	size_t size;
	size_t input_dimension;
	void (*logistic)(struct layer *);
} MLP_layer;


typedef struct mlp{
	MLP_layer *head;
	MLP_layer *tail;
	int guess;
	size_t num_params;
	float learning_rate;
	float *params;
	float *_outputs;
	float (*cost)(struct mlp *);
} MLP;

MLP mlp_from_arr(size_t arr[], size_t size);
MLP load_mlp(const char *filename);

void mlp_forward(MLP *, float *);

//void gradients_wrt_outputs(MLP_layer *);
//void mutate(MLP_layer *, float, float);

void saveMLPToFile(MLP *n, char* filename);

float descend(MLP *n, float *, float *);
float backpropagate(MLP *n, float *);

//int bestGuess(MLP *n);

//These are activation functions
void hypertan(MLP_layer* layer); //Sometimes unstable
void sigmoid(MLP_layer* layer);
void softmax(MLP_layer* layer);
//void relu(MLP_layer* layer); //not stable, be careful
//void leaky_relu(MLP_layer* layer); //not stable, be careful

void dealloc_network(MLP *);

//void printOutputs(MLP_layer *layer);
//void prettyprint(MLP_layer *layer);
//void printActivationGradients(MLP_layer *layer);
//void printWeights(MLP_layer *layer);

#endif
