/* Author: Jonah Siekmann
 * 10/3/2018
 * This is an attempt to write a simple Long Short-Term Memory network (LSTM_layer) implementation. 
 */

#include "LSTM_layer.h"
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

static Gate createGate(float *weights, float bias, size_t sequence_length, size_t input_dimension){
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

LSTM_layer createLSTM_layer(size_t input_dim, size_t size){
	LSTM_layer n;
	n.input_dimension = input_dim + size;
	n.plasticity = 0.05;

	Cell *cells = (Cell*)malloc(size*sizeof(Cell));
	for(int i = 0; i < size; i++){
		Cell *cell = &cells[i];
		
		//allocate weights for this cell
		float *input_nonl_weights = ALLOCATE(float, n.input_dimension);
		float *input_gate_weights = ALLOCATE(float, n.input_dimension);
		float *forget_gate_weights = ALLOCATE(float, n.input_dimension);
		float *output_gate_weights = ALLOCATE(float, n.input_dimension);

		//randomly initialize weights
		for(int j = 0; j < n.input_dimension; j++){
			input_nonl_weights[j] = ((float)(rand()%7000)-3500)/10000;
			input_gate_weights[j] = ((float)(rand()%7000)-3500)/10000;
			forget_gate_weights[j] = ((float)(rand()%7000)-3500)/10000;
			output_gate_weights[j] = ((float)(rand()%7000)-3500)/10000;
		}
		
		//randomly initialize biases
		float input_nonl_bias = ((float)(rand()%7000)-3500)/10000;
		float input_gate_bias = ((float)(rand()%7000)-3500)/10000;
		float forget_gate_bias = ((float)(rand()%7000)-3500)/10000;
		float output_gate_bias = ((float)(rand()%7000)-3500)/10000;

		//Allocate gates
		cell->input_nonl = createGate(input_nonl_weights, input_nonl_bias, UNROLL_LENGTH, n.input_dimension);
		cell->input_gate = createGate(input_gate_weights, input_gate_bias, UNROLL_LENGTH, n.input_dimension);
		cell->forget_gate = createGate(forget_gate_weights, forget_gate_bias, UNROLL_LENGTH, n.input_dimension);
		cell->output_gate = createGate(output_gate_weights, output_gate_bias, UNROLL_LENGTH, n.input_dimension);

		cell->output = ALLOCATE(float, UNROLL_LENGTH);
		cell->state = ALLOCATE(float, UNROLL_LENGTH);
		cell->dstate = ALLOCATE(float, UNROLL_LENGTH);
		cell->gradient = ALLOCATE(float, UNROLL_LENGTH);
		cell->dOutput = ALLOCATE(float, UNROLL_LENGTH);

	}
	n.cells = cells;
	n.size = size;
	n.inputs = ALLOCATE(float*, UNROLL_LENGTH);
	for(int t = 0; t < UNROLL_LENGTH; t++) n.inputs[t] = ALLOCATE(float, n.input_dimension);
	n.input_gradients = ALLOCATE(float*, UNROLL_LENGTH);
	for(int t = 0; t < UNROLL_LENGTH; t++) n.input_gradients[t] = ALLOCATE(float, n.input_dimension);

	reset_inputs(n.inputs, UNROLL_LENGTH, n.input_dimension);
	reset_inputs(n.input_gradients, UNROLL_LENGTH, n.input_dimension);
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

	
float step(LSTM_layer *n, float *input, float *desired){
	if(n->t >= UNROLL_LENGTH || desired == NULL){ //We've reached the max unroll length, so time to do bptt
		//bptt
		printf("doing bptt\n");

		//do gradient math
		for(int t = n->t-1; t >= 0; t--){
//			printf("BACKPROP: doing timestep %d\n", t);
			for(int j = 0; j < n->size; j++){
				Cell *c = &n->cells[j];
				Gate *a = &c->input_nonl;
				Gate *i = &c->input_gate;
				Gate *f = &c->forget_gate;
				Gate *o = &c->output_gate;

				float delta_out;
				float next_dstate;
				float next_forget;

				if(t >= UNROLL_LENGTH-1){
					delta_out = 0; //Zero because no future timesteps
					next_dstate = 0;
					next_forget = 0;
				}else{
					delta_out = n->input_gradients[t+1][n->input_dimension - n->size + j];
					next_dstate = c->dstate[t+1];
					next_forget = f->output[t+1];
				}

				c->dOutput[t] = c->gradient[t] + delta_out;
				c->dstate[t] = c->dOutput[t] * o->output[t] * d_hypertan_element(c->state[t]) + next_dstate * next_forget;

				a->gradient[t] = c->dstate[t] * i->output[t] * (1 - a->output[t] * a->output[t]);
				i->gradient[t] = c->dstate[t] * a->output[t] * i->output[t] * (1 - i->output[t]);
				if(t) f->gradient[t] = c->dstate[t] * c->state[t-1] * f->output[t] * (1 - f->output[t]);
				else  f->gradient[t] = 0;
				o->gradient[t] = c->dOutput[t] * hypertan_element(c->state[t]) * o->output[t] * (1 - o->output[t]);
/*
				printf("BACKPROP t=%d: made d_state from %6.5f * %6.5f * %6.5f + %6.5f * %6.5f\n", t, c->dOutput[t], o->output[t], d_hypertan_element(c->state[t]), next_dstate, next_forget);
				printf("BACKPROP t=%d:		d_state: %6.5f\n", t, c->dstate[t]);
				printf("BACKPROP t=%d: 		d_a: %6.5f\n", t, a->gradient[t]);
				printf("BACKPROP t=%d:		d_i: %6.5f\n", t, i->gradient[t]);
				printf("BACKPROP t=%d:		d_f: %6.5f\n", t, f->gradient[t]);
				printf("BACKPROP t=%d:		d_o: %6.5f\n", t, o->gradient[t]);
*/
				for(int k = 0; k < n->input_dimension; k++){
					n->input_gradients[t][k] += a->gradient[t] * a->weights[k];
					n->input_gradients[t][k] += i->gradient[t] * i->weights[k];
					n->input_gradients[t][k] += f->gradient[t] * f->weights[k];
					n->input_gradients[t][k] += o->gradient[t] * o->weights[k];
//					printf("BACKPROP t=%d: n->input_gradients[%d][%d] set to %f\n", t, t, k, n->input_gradients[t][k]);
				}
//				printf("BACKPROP t=%d: dOutput is %6.5f from gradient %6.5f and dout %6.5f\n", t, c->dOutput[t], c->gradient[t], delta_out); 
			}
		}
		//do the parameter nudges
		for(int t = 0; t < UNROLL_LENGTH; t++){
			for(int j = 0; j < n->size; j++){
				Cell *c = &n->cells[j];
				Gate *a = &c->input_nonl;
				Gate *i = &c->input_gate;
				Gate *f = &c->forget_gate;
				Gate *o = &c->output_gate;
				size_t recurrent_offset = n->input_dimension - n->size;
				for(int k = 0; k < recurrent_offset; k++){
//					printf("BACKPROP: adding %6.5f * %6.5f (%6.5f) to weight %d of a\n", a->gradient[t], n->inputs[t][k], a->gradient[t] * n->inputs[t][k], k);
					a->weights[k] += a->gradient[t] * n->inputs[t][k] * n->plasticity;
					i->weights[k] += i->gradient[t] * n->inputs[t][k] * n->plasticity;
					f->weights[k] += f->gradient[t] * n->inputs[t][k] * n->plasticity;
					o->weights[k] += o->gradient[t] * n->inputs[t][k] * n->plasticity;
				}
				if(t < UNROLL_LENGTH-1){
					for(int k = recurrent_offset; k < n->input_dimension; k++){
//						printf("BACKPROP: 		making a weight adjustment with a.grad[%d+1]: %6.5f * c.output[%d]: %6.5f\n", t, a->gradient[t+1], t, c->output[t]);
						a->weights[k] += a->gradient[t+1] * c->output[t] * n->plasticity;
						i->weights[k] += i->gradient[t+1] * c->output[t] * n->plasticity;
						f->weights[k] += f->gradient[t+1] * c->output[t] * n->plasticity;
						o->weights[k] += o->gradient[t+1] * c->output[t] * n->plasticity;
					}
				}
				a->bias += a->gradient[t] * n->plasticity;
				i->bias += i->gradient[t] * n->plasticity;
				f->bias += f->gradient[t] * n->plasticity;
				o->bias += o->gradient[t] * n->plasticity;
			}
		}
//		printf("weight 0: %6.5f\n", n->cells[0].input_gate.weights[0]);
//		printf("weight 1: %6.5f\n", n->cells[0].input_gate.weights[1]);
//		printf("weight 2: %6.5f\n", n->cells[0].input_gate.weights[2]);
//		printf("bias: %6.5f\n", n->cells[0].input_gate.bias);
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

	//feedforward
	printf("FEEDFORWARD t=%lu: inputs: [", t);
	for(int j = 0; j < n->input_dimension; j++){
		printf("%2.3f", n->inputs[t][j]);
		if(j < n->input_dimension-1) printf(", ");
		else printf("]\n");
	}
	for(int j = 0; j < n->size; j++){
		Cell *c = &n->cells[j];
		Gate *a = &c->input_nonl;
		Gate *i = &c->input_gate;
		Gate *f = &c->forget_gate;
		Gate *o = &c->output_gate;

		a->output[t] = hypertan_element(inner_product(a->weights, n->inputs[t], n->input_dimension) + a->bias);
		i->output[t] = sigmoid_element(inner_product(i->weights, n->inputs[t], n->input_dimension) + i->bias);
		f->output[t] = sigmoid_element(inner_product(f->weights, n->inputs[t], n->input_dimension) + f->bias);
		o->output[t] = sigmoid_element(inner_product(o->weights, n->inputs[t], n->input_dimension) + o->bias);


		a->dOutput[t] = 1 - a->output[t] * a->output[t];
		i->dOutput[t] = i->output[t] * (1 - i->output[t]);
		f->dOutput[t] = f->output[t] * (1 - f->output[t]);
		o->dOutput[t] = o->output[t] * (1 - o->output[t]);

		if(t) c->state[t] = a->output[t] * i->output[t] + f->output[t] * c->state[t-1];
		else c->state[t] = a->output[t] * i->output[t];
		c->output[t] = hypertan_element(c->state[t]) * o->output[t];
		printf("FEEDFORWARD t=%lu: CELL %d: a: %6.5f, i: %6.5f, f: %6.5f, o: %6.5f, state %6.5f\n", t, j, a->output[t], i->output[t], f->output[t], o->output[t], c->state[t]);
	}
	float cost = 0;
	for(int j = 0; j < n->size; j++){
		n->cells[j].gradient[t] = desired[j] - n->cells[j].output[t];
//		printf("FEEDFORWARD t=%lu: Giving cell %d a gradient of %6.5f from %6.5f - %6.5f\n", t, j, n->cells[j].gradient[t], n->cells[j].output[t], desired[j]);
		cost += 0.5 * (desired[j] - n->cells[j].output[t]) * (desired[j] - n->cells[j].output[t]); //L2 loss
	}
	//end feedforward
	printf("FEEDFORWARD t=%lu: output: [", t);
	for(int j = 0; j < n->size; j++){
		printf("%2.3f", n->cells[j].output[t]);
		if(j < n->size-1) printf(", ");
		else printf("]\n");
	}

	n->t++;
	return cost;
}


/*
static float cost(LSTM_layer *n, int label){
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
float backpropagate_cells(LSTM_layer *n, int label){
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
