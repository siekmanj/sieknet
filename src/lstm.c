/* Author: Jonah Siekmann
 * 10/3/2018
 * This is a simple implementation of a Long Short-Term Memory (LSTM) network, as described by Sepp Hochreiter and Jurgen Schmidhuber in their 1993 paper, with the addition of the forget gate described by Felix Gers.
 * Big thanks to Aidan Gomez for his wonderful numerical example of the backpropagation algorithm for an LSTM cell:
 * https://blog.aidangomez.ca/2016/04/17/Backpropogating-an-LSTM-A-Numerical-Example/
 */

#include "lstm.h"
#include <math.h>
#include <string.h>

#define PRINTLIST(name, len) printf("printing %s: [", #name); for(int xyz = 0; xyz < len; xyz++){printf("%5.4f", name[xyz]); if(xyz < len-1) printf(", "); else printf("]\n");}
#define ALLOCATE(TYPE, NUM) (TYPE*)malloc((NUM) * (sizeof(TYPE)));
#define DEBUG 0
#define MAX_GRAD 10


/*
 * Used to initialize input/forget/output gates
 */
static Gate createGate(float *weights, float *bias, size_t sequence_length){
	Gate g;
	g.output = ALLOCATE(float, sequence_length);
	g.dOutput = ALLOCATE(float, sequence_length);
	g.gradient = ALLOCATE(float, sequence_length);

	g.weights = weights;
	g.bias = bias;

	return g;
}

/*
 * Used to reset the hidden state of the lstm
 */ 
void wipe(LSTM *n){
	for(int i = 0; i < n->depth; i++){
		LSTM_layer *l = &n->layers[i];
		for(long t = 0; t < MAX_UNROLL_LENGTH; t++){
			for(long j = 0; j < l->size; j++){
				l->output[t][i] = 0;
				l->cells[j].dOutput[t] = 0;
				l->cells[j].state[t] = 0;
				l->cells[j].dstate[t] = 0;
				l->cells[j].lstate = 0;
				l->cells[j].loutput = 0;
			}
		}
		zero_2d_arr(l->input_gradient, MAX_UNROLL_LENGTH, l->input_dimension);
	}
	zero_2d_arr(n->network_gradient, MAX_UNROLL_LENGTH, n->layers[n->depth-1].size);
	n->t = 0;
}	

float lstm_cost(LSTM *n, float *y){
	MLP *mlp = &n->output_layer;
	float c = n->cost_fn(mlp->output, y, mlp->cost_gradient, n->output_dimension);
	mlp_backward(mlp);

	float *grads = mlp->layers[0].gradient;
	
	size_t t = n->t;
	if(mlp->layers[0].input_dimension != n->layers[n->depth-1].size){
		printf("ERROR: cost_wrapper(): Size mismatch between mlp output layer and final lstm layer (%lu vs %lu)\n", mlp->layers[0].input_dimension, n->layers[n->depth-1].size);
		exit(1);
	}
	for(int i = 0; i < mlp->layers[0].input_dimension; i++)
		n->network_gradient[t][i] = grads[i];

	n->t++;
	return c;
}


/*
 * Used to initialize & allocate memory for a layer of LSTM cells
 */
LSTM_layer create_LSTM_layer(size_t input_dim, size_t size, float *param_addr){
	LSTM_layer l;
	l.input_dimension = input_dim + size;

	Cell *cells = (Cell*)malloc(size*sizeof(Cell));

	int param_idx = 0;
	for(long i = 0; i < size; i++){
		Cell *cell = &cells[i];

		float *input_nonl_bias = &param_addr[param_idx];
		float *input_nonl_weights = &param_addr[param_idx+1];
		xavier_init(&param_addr[param_idx], (l.input_dimension+1), size);
		param_idx += l.input_dimension+1;

		float *input_gate_bias = &param_addr[param_idx];
		float *input_gate_weights = &param_addr[param_idx+1];
		xavier_init(&param_addr[param_idx], (l.input_dimension+1), size);
		param_idx += l.input_dimension+1;
		
		float *forget_gate_bias = &param_addr[param_idx];
		float *forget_gate_weights = &param_addr[param_idx+1];
		xavier_init(&param_addr[param_idx], (l.input_dimension+1), size);
		param_idx += l.input_dimension+1;
		
		float *output_gate_bias = &param_addr[param_idx];
		float *output_gate_weights = &param_addr[param_idx+1];
		xavier_init(&param_addr[param_idx], (l.input_dimension+1), size);
		param_idx += l.input_dimension+1;
		
		//Allocate gates
		cell->input_nonl = createGate(input_nonl_weights, input_nonl_bias, MAX_UNROLL_LENGTH);
		cell->input_gate = createGate(input_gate_weights, input_gate_bias, MAX_UNROLL_LENGTH);
		cell->forget_gate = createGate(forget_gate_weights, forget_gate_bias, MAX_UNROLL_LENGTH);
		cell->output_gate = createGate(output_gate_weights, output_gate_bias, MAX_UNROLL_LENGTH);

		cell->state = ALLOCATE(float, MAX_UNROLL_LENGTH);
		cell->dstate = ALLOCATE(float, MAX_UNROLL_LENGTH);
		cell->dOutput = ALLOCATE(float, MAX_UNROLL_LENGTH);
		cell->lstate = 0;
		cell->loutput = 0;

	}
	l.cells = cells;
	l.size = size;
	l.input = ALLOCATE(float*, MAX_UNROLL_LENGTH);
	
	l.output = ALLOCATE(float*, MAX_UNROLL_LENGTH);
	for(int t = 0; t < MAX_UNROLL_LENGTH; t++) 
		l.output[t] = ALLOCATE(float, l.size);

	l.input_gradient = ALLOCATE(float*, MAX_UNROLL_LENGTH);
	for(long t = 0; t < MAX_UNROLL_LENGTH; t++) 
		l.input_gradient[t] = ALLOCATE(float, l.input_dimension);

	zero_2d_arr(l.input_gradient, MAX_UNROLL_LENGTH, l.input_dimension);

	return l;
}

/*
 * Called through a macro which allows a variable number of parameters
 */
LSTM lstm_from_arr(size_t *arr, size_t len){
	LSTM n;
	n.t = 0;
	n.collapse = 0;
	n.stateful = 0;
	n.seq_len = 25;
	n.learning_rate = 0.01;
	n.input_dimension = arr[0];
	n.output_dimension = arr[len-1];
	n.depth = len-2;
	n.cost_fn = cross_entropy_cost;

	if(len < 3){
		printf("ERROR: lstm_from_arr(): must have at least input dim, hidden layer size, and output dim (3 layers), but only %lu provided.\n", len);
		exit(1);
	}
	size_t num_params = 0;
	for(int i = 1; i < len-1; i++){
		num_params += (4*(arr[i-1]+arr[i]+1)*arr[i]); //gate parameters
	}
	num_params += (arr[len-2]+1)*arr[len-1]; //output layer (mlp) params
	n.num_params = num_params;

	n.layers = ALLOCATE(LSTM_layer, len-2);
	n.params = ALLOCATE(float, num_params);
	
	int param_idx = 0;
	for(int i = 1; i < len-1; i++){
		LSTM_layer l = create_LSTM_layer(arr[i-1], arr[i], &n.params[param_idx]);
		param_idx += (4*(arr[i-1]+arr[i]+1))*arr[i];
		n.layers[i-1] = l;
	}	

	//Allocate the 2d array to store the gradients calculated by the output layer
	n.network_gradient = ALLOCATE(float*, MAX_UNROLL_LENGTH);
	for(long i = 0; i < MAX_UNROLL_LENGTH; i++) 
		n.network_gradient[i] = ALLOCATE(float, arr[len-2]);
	
	n.network_input = ALLOCATE(float*, MAX_UNROLL_LENGTH);
	for(int i = 0; i < MAX_UNROLL_LENGTH; i++)
		n.network_input[i] = ALLOCATE(float, arr[0]);

	zero_2d_arr(n.network_gradient, MAX_UNROLL_LENGTH, arr[len-2]);
	zero_2d_arr(n.network_input, MAX_UNROLL_LENGTH, arr[0]);

	MLP output_mlp;
	output_mlp.layers = ALLOCATE(MLP_layer, sizeof(MLP_layer));
	output_mlp.depth = 1;
	output_mlp.num_params = (arr[len-2]+1)*arr[len-1];
	output_mlp.input_dimension = arr[len-2];
	output_mlp.output_dimension = arr[len-1];
	output_mlp.learning_rate = n.learning_rate/10.0;
	output_mlp.params = &n.params[param_idx];
	output_mlp.output = ALLOCATE(float, arr[len-1]);
	output_mlp.cost_gradient = ALLOCATE(float, arr[len-1]);
	output_mlp.cost_fn = cross_entropy_cost;
	output_mlp.layers[0] = create_MLP_layer(arr[len-2], arr[len-1], output_mlp.params, softmax);
	output_mlp.guess = 0;
	output_mlp.output = output_mlp.layers[0].output;
	n.output_layer = output_mlp;
	
	return n;
}

/*
 * Does elementwise tanh, doesn't return NaN's
 */
float hypertan_element(float x){
	if(x > 7.0)  return 0.999998;
	if(x < -7.0) return -0.999998;
	return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}
/*
 * Returns the derivative of tanh
 */
float d_hypertan_element(float x){
	float x_sq = hypertan_element(x);
	return 1 - x_sq * x_sq;
}

/*
 * Does elementwise sigmoid
 */
float sigmoid_element(float x){
 return 1/(1 + exp(-x));
}

/*
 * Computes the forward pass of a single layer
 */
void lstm_layer_forward(LSTM_layer *l, float *input, size_t t){
	size_t recurrent_offset = l->input_dimension - l->size;

	l->input[t] = input; //save pointer to input for backward pass
	float tmp[l->input_dimension];
	for(int i = 0; i < recurrent_offset; i++)
		tmp[i] = input[i];
	for(int i = recurrent_offset; i < l->input_dimension; i++)
		tmp[i] = l->cells[i - recurrent_offset].loutput;

	//Do output calculations for every cell in this layer
	for(long j = 0; j < l->size; j++){
		Cell *c = &l->cells[j];
		Gate *a = &c->input_nonl;
		Gate *i = &c->input_gate;
		Gate *f = &c->forget_gate;
		Gate *o = &c->output_gate;

		a->output[t] = hypertan_element(inner_product(a->weights, tmp, l->input_dimension) + *a->bias); //input nonlinearity uses hypertangent
		i->output[t] = sigmoid_element(inner_product(i->weights, tmp, l->input_dimension) + *i->bias); //all of the gates use sigmoid
		f->output[t] = sigmoid_element(inner_product(f->weights, tmp, l->input_dimension) + *f->bias);
		o->output[t] = sigmoid_element(inner_product(o->weights, tmp, l->input_dimension) + *o->bias);

		a->dOutput[t] = 1 - a->output[t] * a->output[t];
		i->dOutput[t] = i->output[t] * (1 - i->output[t]);
		f->dOutput[t] = f->output[t] * (1 - f->output[t]);
		o->dOutput[t] = o->output[t] * (1 - o->output[t]);

		c->state[t] = a->output[t] * i->output[t] + f->output[t] * c->lstate; //Calculate the internal cell state
		l->output[t][j] = hypertan_element(c->state[t]) * o->output[t]; //Calculate the output of the cell

#if DEBUG
		if(isnan(a->output[t])){
			printf("ERROR: lstm_layer_forward(): c[%d], a->output[%d] is nan from tanh(%6.5f + %6.5f)\n", j, t, inner_product(a->weights, l->input[t], l->input_dimension), a->bias);
			printf("from inner product of:\n");
			for(int i = 0; i < l->input_dimension; i++){
				printf("%6.5f * %6.5f +\n", a->weights[i], l->input[t][i]);
			}
			exit(1);
		}
		if(isnan(c->state[t])){
			printf("ERROR: lstm_layer_forward(): nan while doing c[%d], state[%d] = %6.5f * %6.5f + %6.5f * %6.5f\n", j, t, a->output[t], i->output[t], f->output[t], c->lstate);
			exit(1);
		}
		if(isnan(l->output[t][j])){
			printf("ERROR: lstm_layer_forward(): c[%d]->output[%d] is nan from tanh(%6.5f * %6.5f)\n", j, t, c->state[t], o->output[t]);
			printf("                      : made %6.5f from %6.5f * %6.5f + %6.5f * %6.5f\n", c->state[t], a->output[t], i->output[t], f->output[t], c->lstate);
			exit(1);
		}
		if(c->state[t] > 1000 || c->state[t] < -1000){
			printf("WARNING: lstm_layer_forward(): c[%d]->state[%d] (%6.5f) is unusually large and may lead to exploding gradients!\n", j, t, c->state[t]);
			printf("                      : made %6.5f from %6.5f * %6.5f + %6.5f * %6.5f\n", c->state[t], a->output[t], i->output[t], f->output[t], c->lstate);
		}
#endif
		c->lstate = c->state[t]; //Set the last timestep's cell state to the current one for the next timestep
		c->loutput = l->output[t][j];
	}
}

/*
 * Does a forward pass through the network
 */
void lstm_forward(LSTM *n, float *x){
	size_t t = n->t;
	for(int i = 0; i < n->input_dimension; i++){
		n->network_input[t][i] = x[i];
	}
	//Feedforward through all LSTM layers
	float *input = n->network_input[t];
	for(int i = 0; i < n->depth; i++){
		LSTM_layer *l = &n->layers[i];
		lstm_layer_forward(l, input, t);
		input = l->output[t];
	}

	//Feedforward through final MLP layer
	mlp_forward(&n->output_layer, input);
	n->guess = n->output_layer.guess;
}

/*
 * Propagate gradients from cost function throughout network, for all timesteps.
 */
void lstm_propagate_gradients(LSTM *n, float **network_grads){
	size_t MAX_TIME = n->t-1;
	float **gradients = network_grads;
	for(int i = n->depth-1; i >= 0; i--){
		LSTM_layer *l = &n->layers[i];
		int recurrent_offset = l->input_dimension - l->size;
		for(long t = MAX_TIME; t >= 0; t--){
			for(long j = 0; j < l->size; j++){
				Cell *c = &l->cells[j];
				Gate *a = &c->input_nonl;
				Gate *i = &c->input_gate;
				Gate *f = &c->forget_gate;
				Gate *o = &c->output_gate;

				float delta_out;
				float next_dstate;
				float next_forget;

				if(t >= MAX_TIME){
					delta_out = 0; //Zero because no future timesteps
					next_dstate = 0;
					next_forget = 0;
				}else{
					delta_out = l->input_gradient[t+1][recurrent_offset + j];
					next_dstate = c->dstate[t+1];
					next_forget = f->output[t+1];
				}
				float grad = gradients[t][j]; 
				c->dOutput[t] = grad + delta_out;
				c->dstate[t] = c->dOutput[t] * o->output[t] * d_hypertan_element(c->state[t]) + next_dstate * next_forget;

				a->gradient[t] = c->dstate[t] * i->output[t] * a->dOutput[t];
				i->gradient[t] = c->dstate[t] * a->output[t] * i->dOutput[t];
				if(t) f->gradient[t] = c->dstate[t] * c->state[t-1] * f->dOutput[t];
				else  f->gradient[t] = 0;
				o->gradient[t] = c->dOutput[t] * hypertan_element(c->state[t]) * o->output[t] * (1 - o->output[t]);

				for(long k = 0; k < l->input_dimension; k++){
					l->input_gradient[t][k] += a->gradient[t] * a->weights[k];
					l->input_gradient[t][k] += i->gradient[t] * i->weights[k];
					l->input_gradient[t][k] += f->gradient[t] * f->weights[k];
					l->input_gradient[t][k] += o->gradient[t] * o->weights[k];
				}
			}
			//clip gradients that are too large (to put a band-aid on exploding gradients)
			for(long j = 0; j < l->input_dimension; j++){
				if(l->input_gradient[t][j] > MAX_GRAD) l->input_gradient[t][j] = MAX_GRAD;
				if(l->input_gradient[t][j] < -MAX_GRAD) l->input_gradient[t][j] = -MAX_GRAD;
			}
		}
		gradients = l->input_gradient; //pass gradients further back through network
	}
}

/*
 * Perform parameter update step for a single layer
 */
void lstm_layer_backward(LSTM_layer *l, size_t max_time, float learning_rate){
	int recurrent_offset = l->input_dimension - l->size;
	for(long t = 0; t <= max_time; t++){
		for(long j = 0; j < l->size; j++){
			Cell *c = &l->cells[j];
			Gate *a = &c->input_nonl;
			Gate *i = &c->input_gate;
			Gate *f = &c->forget_gate;
			Gate *o = &c->output_gate;
			for(long k = 0; k < recurrent_offset; k++){
				a->weights[k] += a->gradient[t] * l->input[t][k] * learning_rate;
				i->weights[k] += i->gradient[t] * l->input[t][k] * learning_rate;
				f->weights[k] += f->gradient[t] * l->input[t][k] * learning_rate;
				o->weights[k] += o->gradient[t] * l->input[t][k] * learning_rate;
			}
			if(t < max_time){
				for(long k = recurrent_offset; k < l->input_dimension; k++){
					a->weights[k] += a->gradient[t+1] * l->output[t][j] * learning_rate;
					i->weights[k] += i->gradient[t+1] * l->output[t][j] * learning_rate;
					f->weights[k] += f->gradient[t+1] * l->output[t][j] * learning_rate;
					o->weights[k] += o->gradient[t+1] * l->output[t][j] * learning_rate;
					if(isnan(a->weights[k])){
						printf("ERROR: layer_backward(): nan'ed a weight while doing nudge, from %6.5f * %6.5f * %6.5f\n", a->gradient[t+1], l->output[t][j], learning_rate);
						exit(1);
					}
				}
			}
			*a->bias += a->gradient[t] * learning_rate;
			*i->bias += i->gradient[t] * learning_rate;
			*f->bias += f->gradient[t] * learning_rate;
			*o->bias += o->gradient[t] * learning_rate;
		}
	}
}

/*
 * Performs parameter update for all LSTM layers
 */
void lstm_backward(LSTM *n){
	if(n->t >= n->seq_len){
		lstm_propagate_gradients(n, n->network_gradient);
		for(int i = n->depth-1; i >= 0; i--){
			lstm_layer_backward(&n->layers[i], n->t-1, n->learning_rate);
		}
		for(int i = 0; i < n->depth; i++){
			LSTM_layer *l = &n->layers[i];
			zero_2d_arr(l->output, MAX_UNROLL_LENGTH, l->size);
			zero_2d_arr(l->input_gradient, MAX_UNROLL_LENGTH, l->input_dimension);
		}
		if(!n->stateful) wipe(n);
		n->t = 0;
	}
}


void dealloc_lstm(LSTM *n){
	for(int i = 0; i < n->depth; i++){
		LSTM_layer *l = &n->layers[i];
		for(int j = 0; j < l->size; j++){
			Cell *c = &l->cells[j];
			Gate *a = &c->input_nonl;
			Gate *i = &c->input_gate;
			Gate *f = &c->forget_gate;
			Gate *o = &c->output_gate;
			free(a->output);
			free(a->dOutput);
			free(a->gradient);
			
			free(i->output);
			free(i->dOutput);
			free(i->gradient);
			
			free(f->output);
			free(f->dOutput);
			free(f->gradient);
			
			free(o->output);
			free(o->dOutput);
			free(o->gradient);

			free(c->state);
			free(c->dstate);
			free(c->dOutput);
		}
		for(int t = 0; t < MAX_UNROLL_LENGTH; t++){
			free(l->input_gradient[t]);
			free(l->output[t]);
		}
		free(l->input_gradient);
		free(l->output);
		free(l->cells);
	}
	for(int t = 0; t < MAX_UNROLL_LENGTH; t++){
		free(n->network_gradient[t]);
		free(n->network_input[t]);
	}
	free(n->network_gradient);
	free(n->network_input);
	free(n->layers);
	free(n->params);
	free(n->output_layer.layers[0].output);
	free(n->output_layer.layers[0].gradient);
	free(n->output_layer.layers[0].neurons);
	free(n->output_layer.layers);
	free(n->output_layer.output);
	free(n->output_layer.cost_gradient);
}
 /*
  * IO FUNCTIONS FOR READING AND WRITING TO A FILE
  */

static void getWord(FILE *fp, char* dest){
  memset(dest, '\0', strlen(dest));
  int res = fscanf(fp, " %1023s", dest);
}

void save_lstm(LSTM *n, const char *filename){
	FILE *fp = fopen(filename, "w");

	fprintf(fp, "LSTM %lu %lu ", n->depth, n->input_dimension);
	for(int i = 0; i < n->depth; i++){
		fprintf(fp, "%lu", n->layers[i].size);
		fprintf(fp, " ");
	}
	fprintf(fp, "%lu\n", n->output_layer.layers[0].size);

	for(int i = 0; i < n->num_params; i++){
		fprintf(fp, "%f", n->params[i]);
		if(i < n->num_params-1) fprintf(fp, " ");
		else fprintf(fp, "\n");
	}
	fclose(fp);
}

LSTM load_lstm(const char *filename){
  FILE *fp = fopen(filename, "rb");
  char buff[1024];
	int f;
  memset(buff, '\0', 1024);

  getWord(fp, buff); //Get first word to check if MLP file

  if(strcmp(buff, "LSTM") != 0){
    printf("ERROR: [%s] is not an LSTM.\n", buff);
    exit(1);
  }
	size_t num_layers, input_dim;
	f = fscanf(fp, "%lu %lu", &num_layers, &input_dim);
	size_t arr[num_layers+2];
	arr[0] = input_dim;
	for(int i = 1; i <= num_layers; i++){
		f = fscanf(fp, " %lu", &arr[i]);
	}
	f = fscanf(fp, " %lu", &arr[num_layers+1]);
	LSTM n = lstm_from_arr(arr, num_layers+2);
	for(int i = 0; i < n.num_params; i++){
		f = fscanf(fp, "%f", &n.params[i]);
	}
	return n;
}
