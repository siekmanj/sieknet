#ifndef LSTM_H
#define LSTM_H

#include <stdlib.h>
#include <stdio.h>
#include "MLP.h"


//typedef MLP LSTM;
typedef struct cell{
	Neuron input_activation;
	Neuron input_gate;
	Neuron forget_gate;
	Neuron output_gate;
	float state;
	float output;
	float dActivation;
	float gradient;
} Cell;

typedef struct lstm_layer{
	Cell *cells;
	float *last_out;
	size_t input_dimension;
	size_t size;
	double plasticity;
//	struct Layer *input_layer;
//	struct Layer *output_layer;
//	void (squish)(struct Layer*);
} LSTM;

LSTM createLSTM(size_t input_dimension, size_t cells);
//LSTM loadLSTMFromFile(const char *filename);

void feedforward_forget(LSTM *n, float *);
void process_cell(Cell *c);
//float step(LSTM *n, int label;

#endif
