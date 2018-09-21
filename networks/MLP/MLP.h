#ifndef MLP_H
#define MLP_H

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
	float activationGradient;
} Neuron;

typedef struct Layer{
  Neuron *neurons;
  void *input_layer;
	void *output_layer;
	void (*squish)(void*);
	size_t size;
} Layer;


typedef struct MLP{
	Layer *input;
	Layer *output;
	//void (*setInputs)(void* layer, void* arr);
	double performance;
	double plasticity;
	unsigned long age;
} MLP;

MLP mlp_from_arr(size_t arr[], size_t size);
MLP loadMLPFromFile(const char *filename);

void addLayer(MLP *n, size_t size);
void setInputs(MLP *n, float* arr);
void calculate_outputs(Layer*);
void feedforward(MLP *n);
void saveMLPToFile(MLP *n, char* filename);

float descend(MLP *n, int label);
float backpropagate(Layer *output_layer, int label, float plasticity);

int bestGuess(MLP *n);

void printOutputs(Layer *layer);
void prettyprint(Layer *layer);
void printActivationGradients(Layer *layer);
void printWeights(Layer *layer);

#endif
