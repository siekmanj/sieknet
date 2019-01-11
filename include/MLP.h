#ifndef MLP_H
#define MLP_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

// some magic
#define createMLP(...) mlp_from_arr((size_t[]){__VA_ARGS__}, sizeof((size_t[]){__VA_ARGS__})/sizeof(size_t))

typedef struct Neuron{
	float *weights;
	float input;
	float bias;
	float activation;
	float dActivation;	
	float gradient;
} Neuron;

typedef struct Layer{
	Neuron *neurons;
	struct Layer *input_layer;
	struct Layer *output_layer;
	void (*squish)(struct Layer*);
	float dropout;
	size_t size;
} Layer;


typedef struct MLP{
	Layer *input;
	Layer *output;
	double plasticity;
} MLP;

Layer *create_layer(size_t, Layer*);

MLP mlp_from_arr(size_t arr[], size_t size);
MLP loadMLPFromFile(const char *filename);
MLP initMLP();

void addLayer(MLP *, size_t);
void calculate_inputs(Layer*);
void feedforward(MLP *, float *);

void gradients_wrt_outputs(Layer *);
void mutate(Layer *, float, float);

void saveMLPToFile(MLP *n, char* filename);

float descend(MLP *n, float *, float *);
float backpropagate(MLP *n, float *);

int bestGuess(MLP *n);

//These are activation functions
//You can set these by assigning layer->squish = hypertan/sigmoid/etc
void hypertan(Layer* layer); //Sometimes unstable
void sigmoid(Layer* layer);
void softmax(Layer* layer);
void relu(Layer* layer); //not stable, be careful
void leaky_relu(Layer* layer); //not stable, be careful

void dealloc_network(MLP *);

void printOutputs(Layer *layer);
void prettyprint(Layer *layer);
void printActivationGradients(Layer *layer);
void printWeights(Layer *layer);

#endif
