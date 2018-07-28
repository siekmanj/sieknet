#include <stdlib.h>
#include <stdio.h>
#include <time.h>

typedef struct Neuron{
	float *weights;
	float bias;
	float output;
	float input;
	float dOutput;
	float activationGradient;
} Neuron;

typedef struct Layer{
  Neuron *neurons;
  void *input_layer;
	void *output_layer;
  size_t size;
} Layer;

typedef struct MLP{
	Layer *input;
	Layer *output;
	unsigned long age;
} MLP;

MLP initMLP();
void addLayer(MLP *n, size_t size);
void setInputs(MLP *n, float* arr);
float runEpoch(MLP *n, int label);

void printOutputs(Layer *layer);
void prettyprint(Layer *layer);
void printActivationGradients(Layer *layer);
void printWeights(Layer *layer);
