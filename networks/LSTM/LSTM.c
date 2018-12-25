/* Author: Jonah Siekmann
 * 10/3/2018
 * This is an attempt to write a simple Long Short-Term Memory network (LSTM) implementation. 
 */

#include "LSTM.h"
#include <math.h>
#include <string.h>
//Thanks to Arun Mallya for an excellent writeup on the backprop for an lstm
//http://arunmallya.github.io/writeups/nn/lstm/index.html

/*
 * Description: Initializes a long short-term memory network object.
 */ 
static LSTM initLSTM(){
	LSTM n;
	n.cells = NULL;
	n.last_out = NULL;
	n.input_dimension = 0;
	n.size = 0;
	n.plasticity = 0.01;
	return n;
}

float hypertan_element(float x){
	return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}
float d_hypertan_element(float x){
	float x_sq = hypertan_element(x);
	return 1 - x_sq * x_sq;
}

float sigmoid_element(float x){
 return 1/(1 + exp(-x));
}

float inner_product(float *x, float *y, size_t length){
	float sum = 0;
	for(int i = 0; i < length; i++){
		sum += x[i] * y[i];	
	}
	return sum;
}

	

/*
 * input MUST BE SAME SIZE AS n.input_dimension
 */
void feedforward_forget(LSTM *n, float *input){

	//Create a new input vector containing last timestep's recurrent outputs, concatenate with old input vector
	for(int i = 0; i < n->input_dimension; i++) n->inputs[i] = input[i];
	for(int i = n->input_dimension; i < n->input_dimension + n->size; i++) n->inputs[i] = n->last_out[i-n->input_dimension];

	printf("last output: [");
	for(int j = 0; j < n->size; j++){
		printf("%4.3f", n->last_out[j]);
		if(j < n->size-1) printf(", ");
		else printf("]\n");
	}

	printf("concatenated inputs: [");
	for(int j = 0; j < (n->size + n->input_dimension); j++){
		printf("%4.3f", n->inputs[j]);
		if(j < (n->size + n->input_dimension)-1) printf(", ");
		else printf("]\n");
	}

	for(int i = 0; i < n->size; i++){
		Cell *c = &n->cells[i];
		Neuron *inpt_actv = &c->input_activation;
		Neuron *inpt_gate = &c->input_gate;
		Neuron *frgt_gate = &c->forget_gate;
		Neuron *otpt_gate = &c->output_gate;

	 	inpt_actv->input = inner_product(inpt_actv->weights, n->inputs, n->input_dimension) + inpt_actv->bias;
		inpt_gate->input = inner_product(inpt_gate->weights, n->inputs, n->input_dimension) + inpt_gate->bias;
		frgt_gate->input = inner_product(frgt_gate->weights, n->inputs, n->input_dimension) + frgt_gate->bias;
		otpt_gate->input = inner_product(otpt_gate->weights, n->inputs, n->input_dimension) + otpt_gate->bias;

		inpt_actv->activation = hypertan_element(inpt_actv->input);
		inpt_gate->activation = sigmoid_element(inpt_gate->input);
		frgt_gate->activation = sigmoid_element(frgt_gate->input);
		otpt_gate->activation = sigmoid_element(otpt_gate->input);

		inpt_actv->dActivation = 1 - inpt_actv->activation * inpt_actv->activation;
		inpt_gate->dActivation = inpt_gate->activation * (1 - inpt_gate->activation);
		frgt_gate->dActivation = frgt_gate->activation * (1 - frgt_gate->activation);
		otpt_gate->dActivation = otpt_gate->activation * (1 - otpt_gate->activation);

		c->lstate = c->state;
		c->state = inpt_actv->activation * inpt_gate->activation + frgt_gate->activation * c->state;
		c->output = hypertan_element(c->state) * otpt_gate->activation;
		n->last_out[i] = c->output;
		printf("	cell %d output was %5.3f, last_out[%d] is now %5.3f, input act of %5.3f, input gate of %5.3f, forget gate of %5.3f, output gate of %5.3f\n", i, c->output, i,  n->last_out[i], inpt_actv->activation, inpt_gate->activation, frgt_gate->activation, otpt_gate->activation);
	}
}

static float cost(LSTM *n, int label){
	//cost
	float sum = 0;
	for(int i = 0; i < n->size; i++){
		if(i==label){
			n->cells[i].gradient = 1 - n->cells[i].output;
			sum += 0.5 * pow(1 - n->cells[i].output, 2);
		}
		else{
			n->cells[i].gradient = 0.5 * pow(n->cells[i].output, 2);
			sum += 0.5 * pow(n->cells[i].output, 2);
		}
	}
	return sum;
}

/*
 * Single-layer backprop for now
 */
float backpropagate_cells(LSTM *n, int label){
	float c = cost(n, label);
	
	for(int i = 0; i < n->size; i++){
		Cell *cell = &n->cells[i];
		//dH = cell->gradient;
//		float delta_output = cell->gradient * cell->output_gate->activation;
		
		
		Neuron *inpt_actv = &cell->input_activation;
		Neuron *inpt_gate = &cell->input_gate;
		Neuron *frgt_gate = &cell->forget_gate;
		Neuron *otpt_gate = &cell->output_gate;

		cell->dstate += cell->gradient * otpt_gate->activation * (1 - pow(hypertan_element(cell->state), 2));

		inpt_actv->gradient = cell->dstate * inpt_gate->activation;
		inpt_gate->gradient = cell->dstate * inpt_actv->activation;
		frgt_gate->gradient = cell->dstate * cell->lstate;
		otpt_gate->gradient = cell->gradient * hypertan_element(cell->state);

		cell->ldstate = cell->dstate * frgt_gate->activation;
		
		//Weight nudges
		for(int j = 0; j < n->input_dimension; j++){
			inpt_actv->weights[j] += inpt_actv->gradient * inpt_actv->dActivation * n->inputs[j];
			inpt_gate->weights[j] += inpt_gate->gradient * inpt_gate->dActivation * n->inputs[j];
			frgt_gate->weights[j] += frgt_gate->gradient * frgt_gate->dActivation * n->inputs[j];
			otpt_gate->weights[j] += otpt_gate->gradient * otpt_gate->dActivation * n->inputs[j];
		}
		inpt_actv->bias += inpt_actv->gradient * inpt_actv->dActivation;
		inpt_gate->bias += inpt_gate->gradient * inpt_gate->dActivation;
		frgt_gate->bias += frgt_gate->gradient * frgt_gate->dActivation;
		otpt_gate->bias += otpt_gate->gradient * otpt_gate->dActivation;
	}
	return c;
}

LSTM createLSTM(size_t size, size_t input_dim){
	LSTM n = initLSTM();
	size_t input_dimension = input_dim + size;
	Cell *cells = (Cell*)malloc(size*sizeof(Cell));
	for(int i = 0; i < size; i++){
		//allocate weights for this cell
		float *input_activation_weights = (float*)malloc(input_dimension*sizeof(float));
		float *input_gate_weights = (float*)malloc(input_dimension*sizeof(float));
		float *forget_gate_weights = (float*)malloc(input_dimension*sizeof(float));
		float *output_gate_weights = (float*)malloc(input_dimension*sizeof(float));
		for(int j = 0; j < input_dimension; j++){
			input_activation_weights[j] = ((float)(rand()%7000)-3500)/10000;
			input_gate_weights[j] = ((float)(rand()%7000)-3500)/10000;
			forget_gate_weights[j] = ((float)(rand()%7000)-3500)/10000;
			output_gate_weights[j] = ((float)(rand()%7000)-3500)/10000;
		}
		cells[i].input_activation.weights = input_activation_weights;
		cells[i].input_gate.weights = input_gate_weights;
		cells[i].forget_gate.weights = forget_gate_weights;
		cells[i].output_gate.weights = output_gate_weights;

		cells[i].input_activation.bias = ((float)(rand()%7000)-3500)/10000;
		cells[i].input_gate.bias = ((float)(rand()%7000)-3500)/10000;
		cells[i].forget_gate.bias = ((float)(rand()%7000)-3500)/10000;
		cells[i].output_gate.bias = ((float)(rand()%7000)-3500)/10000;

		cells[i].state = 0;
		cells[i].output = 0;
		cells[i].dstate = 0.01;
		cells[i].gradient = 0.01;
	
	}
	n.cells = cells;
	n.last_out = (float*)malloc(size * sizeof(float));
	n.size = size;
	n.input_dimension = input_dim;
	n.inputs = (float*)malloc((n.size + n.input_dimension) * sizeof(float));
	return n;
}
