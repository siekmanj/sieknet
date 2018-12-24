/* Author: Jonah Siekmann
 * 10/3/2018
 * This is an attempt to write a simple Long Short-Term Memory network (LSTM) implementation. 
 */

#include "LSTM.h"
#include <math.h>
#include <string.h>

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
	float *tmp = (float*)malloc((n->size + n->input_dimension)*sizeof(float));
	for(int i = 0; i < n->input_dimension; i++) tmp[i] = input[i];
	for(int i = n->input_dimension; i < n->input_dimension + n->size; i++) tmp[i] = n->last_out[i-n->input_dimension];

	printf("last output: [");
	for(int j = 0; j < n->size; j++){
		printf("%4.3f", n->last_out[j]);
		if(j < n->size-1) printf(", ");
		else printf("]\n");
	}

	printf("concatenated inputs: [");
	for(int j = 0; j < (n->size + n->input_dimension); j++){
		printf("%4.3f", tmp[j]);
		if(j < (n->size + n->input_dimension)-1) printf(", ");
		else printf("]\n");
	}

	for(int i = 0; i < n->size; i++){
		Cell *c = &n->cells[i];
		Neuron *inpt_actv = &c->input_activation;
		Neuron *inpt_gate = &c->input_gate;
		Neuron *frgt_gate = &c->forget_gate;
		Neuron *otpt_gate = &c->output_gate;

	 	inpt_actv->input = inner_product(inpt_actv->weights, tmp, n->input_dimension) + inpt_actv->bias;
		inpt_gate->input = inner_product(inpt_gate->weights, tmp, n->input_dimension) + inpt_gate->bias;
		frgt_gate->input = inner_product(frgt_gate->weights, tmp, n->input_dimension) + frgt_gate->bias;
		otpt_gate->input = inner_product(otpt_gate->weights, tmp, n->input_dimension) + otpt_gate->bias;

		inpt_actv->activation = hypertan_element(inpt_actv->input);
		inpt_gate->activation = sigmoid_element(inpt_gate->input);
		frgt_gate->activation = sigmoid_element(frgt_gate->input);
		otpt_gate->activation = sigmoid_element(otpt_gate->input);

		c->state = inpt_actv->activation * inpt_gate->activation + frgt_gate->activation * c->state;
		c->output = hypertan_element(c->state) * otpt_gate->activation;
		n->last_out[i] = c->output;
		printf("	cell %d output was %5.3f, last_out[%d] is now %5.3f, input act of %5.3f, input gate of %5.3f, forget gate of %5.3f, output gate of %5.3f\n", i, c->output, i,  n->last_out[i], inpt_actv->activation, inpt_gate->activation, frgt_gate->activation, otpt_gate->activation);
	}
	free(tmp);
}

float backpropagate(LSTM *n, int label){
	float delta_out = 0;
	float delta_t

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
		cells[i].dActivation = 0.01;
		cells[i].gradient = 0.01;
	
	}
	n.cells = cells;
	n.last_out = (float*)malloc(size * sizeof(float));
	n.size = size;
	n.input_dimension = input_dim;
	return n;
}
