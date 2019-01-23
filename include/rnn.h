#ifndef RNN_H
#define RNN_H

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "MLP.h"

// some magic to allow arbitrary numbers of parameters
#define create_rnn(...) rnn_from_arr((size_t[]){__VA_ARGS__}, sizeof((size_t[]){__VA_ARGS__})/sizeof(size_t))

typedef struct rnn_layer{
	Neuron *neurons;
	float **gradient
	float **output;
	float **input;
	size_t size;
	size_t input_dimension;



} RNN_layer;

RNN rnn_from_arr(size_t arr[], size_t size);
RNN load_rnn(const char *filename);
void save_rnn(RNN *, const char *filename);

void rnn_forward(RNN *, float *);
void rnn_backward(RNN *, float *);

//void feedforward_recurrent(RNN *n);
//void saveRNNToFile(RNN *n, char* filename);

float step(RNN *n, int label);

#endif
