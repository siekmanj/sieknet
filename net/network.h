#include <stdlib.h>
#include <stdio.h>
#include <time.h>

typedef struct Neuron{
	float *weights;
	float bias;
	float output;
	float input;
	float dBias;
	float dOutput;
	float activationGradient;
} Neuron;

typedef struct Layer{
  Neuron *neurons;
  void *input_layer;
	void *output_layer;
  size_t size;
} Layer;

typedef struct Network{
	Layer *input;
	Layer *output;
	unsigned long age;
} Network;

Network initNetwork();
void addLayer(Network *n, size_t size);
void setInputs(Network *n, float* arr);
float runEpoch(Network *n, int label);
Layer *createLayer(size_t size, Layer *previousLayer);
float squish(float input);
void calculateOutputs(Layer *layer);
void setOutputs(Layer *layer, float *outputs);
void printOutputs(Layer *layer);
void prettyprint(Layer *layer);
void printActivationGradients(Layer *layer);
void printWeights(Layer *layer);
void backpropagate(Layer *output_layer, float plasticity);
float cost(Layer*, int);
