#ifndef LSTM_H
#define LSTM_H

#include <stdlib.h>
#include <stdio.h>
#include "MLP.h"


// some magic to allow arbitrary numbers of parameters
#define createLSTM(...) lstm_from_arr((size_t[]){__VA_ARGS__}, sizeof((size_t[]){__VA_ARGS__})/sizeof(size_t))

//typedef MLP LSTM;
typedef struct cell{
	Neuron input_activation;
	Neuron input_gate;
	Neuron forget_gate;
	Neuron output_gate;

} Cell;

typedef struct lstm_layer{
	Cell *cells;
	struct Layer *input_layer;
	struct Layer *output_layer;
	void (squish)(struct Layer*);
} Cell_Layer;

typedef struct lstm{
	Layer *input;
	Layer *output;
	Cell_Layer *hidden;
	double plasticity;
	double performance;
} LSTM;

LSTM createLSTM(size_t input_dimension, size_t cells, size_t output_dimension);
LSTM loadLSTMFromFile(const char *filename);

void feedforward_forget(LSTM *n);
void process_cell(Cell *c);
//float step(LSTM *n, int label;

#endif
