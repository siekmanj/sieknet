#ifndef RNN_H
#define RNN_H

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "MLP.h"

// some magic
#define createRNN(...) rnn_from_arr((size_t[]){__VA_ARGS__}, sizeof((size_t[]){__VA_ARGS__})/sizeof(size_t))


typedef MLP RNN;/*{
	Layer *input;
	Layer *output;
	Layer *recurrent;
	double performance;
	double plasticity;
	unsigned long age;
} RNN;*/

RNN rnn_from_arr(size_t arr[], size_t size);
//RNN loadRNNFromFile(const char *filename);

//void addLayer(RNN *n, size_t size);
//void setInputs(RNN *n, char* arr);
void setOneHotInput(RNN *n, float* arr);
//void feedforward(RNN *n);
//void saveRNNToFile(RNN *n, char* filename);

float step(RNN *n, int label);
//float descend(RNN *n, int label);
//float backpropagate_through_time(Layer *output_layer, int label, float plasticity);

//int bestGuess(RNN *n);


//void printOutputs(Layer *layer);
//void prettyprint(Layer *layer);
//void printActivationGradients(Layer *layer);
//void printWeights(Layer *layer);

#endif
