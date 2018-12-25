/* Author: Jonah Siekmann
 * 10/3/2018
 * This is an attempt to write a simple Long Short-Term Memory network (LSTM) implementation. 
 */

#include "LSTM.h"
#include <math.h>
#include <string.h>

#define ALLOCATE(TYPE, NUM) (TYPE*)malloc(NUM * sizeof(TYPE));

void printlist(float *arr, size_t len){
	printf("[");
	for(int i = 0; i < len; i++){
		printf("%6.5f", arr[i]);
		if(i < len-1) printf(", ");
		else printf("]\n");
	}
}

Gate createGate(float *weights, float *bias, size_t sequence_length){
	Gate g;
	g.output = ALLOCATE(float, sequence_length);
	g.dOutput = ALLOCATE(float, sequence_length);
	g.gradient = ALLOCATE(float, sequence_length);

	g.weights = weights;
	g.bias = bias;

	return g;
}

void reset_inputs(float **arr, size_t sequence_length, size_t input_dimension){
	for(int i = 0; i < sequence_length; i++){
		for(int j = 0; j < input_dimension; j++){
			arr[i][j] = 0.0;
		}
	}
}

//Thanks to Arun Mallya for an excellent writeup on the backprop for an lstm
//http://arunmallya.github.io/writeups/nn/lstm/index.html

LSTM createLSTM(size_t input_dim, size_t size){
	LSTM n;
	n.input_dimension = input_dim + size;

	Cell *cells = (Cell*)malloc(size*sizeof(Cell));
	for(int i = 0; i < size; i++){
		Cell *cell = &cells[i];
		
		//allocate weights for this cell
		float *input_nonl_weights = ALLOCATE(float, n.input_dimension);
		float *input_gate_weights = ALLOCATE(float, n.input_dimension);
		float *forget_gate_weights = ALLOCATE(float, n.input_dimension);
		float *output_gate_weights = ALLOCATE(float, n.input_dimension);

		//ditto for biases
		float *input_nonl_bias = ALLOCATE(float, 1);
		float *input_gate_bias = ALLOCATE(float, 1);
		float *forget_gate_bias = ALLOCATE(float, 1);
		float *output_gate_bias = ALLOCATE(float, 1);
		
		//randomly initialize weights
		for(int j = 0; j < n.input_dimension; j++){
			input_nonl_weights[j] = ((float)(rand()%7000)-3500)/10000;
			input_gate_weights[j] = ((float)(rand()%7000)-3500)/10000;
			forget_gate_weights[j] = ((float)(rand()%7000)-3500)/10000;
			output_gate_weights[j] = ((float)(rand()%7000)-3500)/10000;
		}
		
		//randomly initialize biases
		*input_nonl_bias = ((float)(rand()%7000)-3500)/10000;
		*input_gate_bias = ((float)(rand()%7000)-3500)/10000;
		*forget_gate_bias = ((float)(rand()%7000)-3500)/10000;
		*output_gate_bias = ((float)(rand()%7000)-3500)/10000;

		//Allocate gates
		cell->input_nonl = createGate(input_nonl_weights, input_nonl_bias, UNROLL_LENGTH);
		cell->input_gate = createGate(input_gate_weights, input_gate_bias, UNROLL_LENGTH);
		cell->forget_gate = createGate(forget_gate_weights, forget_gate_bias, UNROLL_LENGTH);
		cell->output_gate = createGate(output_gate_weights, output_gate_bias, UNROLL_LENGTH);

		cell->output = ALLOCATE(float, UNROLL_LENGTH);
		cell->state = ALLOCATE(float, UNROLL_LENGTH);
		cell->dstate = ALLOCATE(float, UNROLL_LENGTH);
		cell->gradient = ALLOCATE(float, UNROLL_LENGTH);

	}
	n.cells = cells;
	n.size = size;
	n.hidden = ALLOCATE(float, n.input_dimension);
	n.inputs = ALLOCATE(float*, UNROLL_LENGTH);
	for(int t = 0; t < UNROLL_LENGTH; t++) n.inputs[t] = ALLOCATE(float, n.input_dimension);
	reset_inputs(n.inputs, UNROLL_LENGTH, n.input_dimension);
	n.t = 0;
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

	
void step(LSTM *n, float *input, float *desired){
	if(n->t >= UNROLL_LENGTH || desired == NULL){ //We've reached the max unroll length, so time to do bptt
		//bptt
		for(int j = 0; j < n->size; j++){
			Cell *c = &n->cells[j];
			Gate *a = &c->input_nonl;
			Gate *i = &c->input_gate;
			Gate *f = &c->forget_gate;
			Gate *o = &c->output_gate;
			for(int t =

		}
		//end bptt
		n->t = 0;
		reset_inputs(n->inputs, UNROLL_LENGTH, n->input_dimension);
	}
	size_t t = n->t;
	for(int i = 0; i < n->input_dimension - n->size; i++){
		n->inputs[t][i] = input[i];
	}
	for(int i = n->input_dimension - n->size; i < n->input_dimension; i++){

		if(t) n->inputs[t][i] = n->cells[i - (n->input_dimension - n->size)].output[t-1]; //recurrent input
		else  n->inputs[t][i] = n->cells[i - (n->input_dimension - n->size)].output[UNROLL_LENGTH-1];
	}
	
	printf("concatenated inputs: [");
	for(int j = 0; j < n->input_dimension; j++){
		printf("%4.3f", n->inputs[t][j]);
		if(j < n->input_dimension-1) printf(", ");
		else printf("]\n");
	}

	//feedforward
	for(int j = 0; j < n->size; j++){
		Cell *c = &n->cells[j];
		Gate *a = &c->input_nonl;
		Gate *i = &c->input_gate;
		Gate *f = &c->forget_gate;
		Gate *o = &c->output_gate;

		a->output[t] = hypertan_element(inner_product(a->weights, n->inputs[t], n->input_dimension) + *a->bias);
		i->output[t] = sigmoid_element(inner_product(i->weights, n->inputs[t], n->input_dimension) + *i->bias);
		f->output[t] = sigmoid_element(inner_product(f->weights, n->inputs[t], n->input_dimension) + *f->bias);
		o->output[t] = sigmoid_element(inner_product(o->weights, n->inputs[t], n->input_dimension) + *o->bias);


		a->dOutput[t] = 1 - a->output[t] * a->output[t];
		i->dOutput[t] = i->output[t] * (1 - i->output[t]);
		f->dOutput[t] = f->output[t] * (1 - f->output[t]);
		o->dOutput[t] = o->output[t] * (1 - o->output[t]);

		if(t) c->state[t] = a->output[t] * i->output[t] + f->output[t] * c->state[t-1];
		else c->state[t] = a->output[t] * i->output[t];
		c->output[t] = hypertan_element(c->state[t]) * o->output[t];
		printf("CELL %d: a_%d: %6.5f, i_%d: %6.5f, f_%d: %6.5f, o_%d: %6.5f, state %6.5f\n", j, t, a->output[t], t, i->output[t], t, f->output[t], t, o->output[t], c->state[t]);
	}
	//end feedforward
	printf("output: [");
	for(int j = 0; j < n->size; j++){
		printf("%4.3f", n->cells[j].output[t]);
		if(j < n->size-1) printf(", ");
		else printf("]\n");
	}

	printf("finished feedforward\n");
	n->t++;
}


/*
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
*/
/*
 * Single-layer backprop for now
 *
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
		
		printf("	Cell %d: state %6.5f, dstate %6.5f, lstate %6.5f, ldstate %6.5f\n", i, cell->state, cell->dstate, cell->lstate, cell->ldstate);
		
		//Weight nudges
		for(int j = 0; j < n->input_dimension; j++){
			inpt_actv->weights[j] += inpt_actv->gradient * inpt_actv->dActivation * n->inputs[j] * n->plasticity;
			inpt_gate->weights[j] += inpt_gate->gradient * inpt_gate->dActivation * n->inputs[j] * n->plasticity;
			frgt_gate->weights[j] += frgt_gate->gradient * frgt_gate->dActivation * n->inputs[j] * n->plasticity;
			otpt_gate->weights[j] += otpt_gate->gradient * otpt_gate->dActivation * n->inputs[j] * n->plasticity;
		}
		inpt_actv->bias += inpt_actv->gradient * inpt_actv->dActivation * n->plasticity;
		inpt_gate->bias += inpt_gate->gradient * inpt_gate->dActivation * n->plasticity;
		frgt_gate->bias += frgt_gate->gradient * frgt_gate->dActivation * n->plasticity;
		otpt_gate->bias += otpt_gate->gradient * otpt_gate->dActivation * n->plasticity;
	}
	return c;
}
*/
