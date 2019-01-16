/* Author: Jonah Siekmann
 * 10/3/2018
 * This is a simple implementation of a Long Short-Term Memory (LSTM) network, as described by Sepp Hochreiter and Jurgen Schmidhuber in their 1993 paper, with the addition of the forget gate described by Felix Gers.
 * Big thanks to Aidan Gomez for his wonderful numerical example of the backpropagation algorithm for an LSTM cell:
 * https://blog.aidangomez.ca/2016/04/17/Backpropogating-an-LSTM-A-Numerical-Example/
 */

#include "LSTM.h"
#include <math.h>
#include <string.h>

#define ALLOCATE(TYPE, NUM) (TYPE*)malloc(NUM * sizeof(TYPE));
#define SOFTMAX 0
#define DEBUG 1
#define MAX_GRAD 10

/*
 * Handy function for zeroing out a 2d array
 */
void zero_2d_arr(float **arr, size_t sequence_length, size_t input_dimension){
	for(long i = 0; i < sequence_length; i++){
		for(long j = 0; j < input_dimension; j++){
			arr[i][j] = 0.0;
		}
	}
}

/*
 * Used to initialize input/forget/output gates
 */
static Gate createGate(float *weights, float bias, size_t sequence_length, size_t input_dimension){
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
	LSTM_layer *l = n->head;
	while(l){
		for(long t = 0; t < MAX_UNROLL_LENGTH; t++){
			for(long i = 0; i < l->size; i++){
				l->cells[i].output[t] = 0;
				l->cells[i].gradient[t] = 0;
				l->cells[i].dOutput[t] = 0;
				l->cells[i].state[t] = 0;
				l->cells[i].dstate[t] = 0;
				l->cells[i].lstate = 0;
				l->cells[i].loutput = 0;
				l->output[i] = 0;
			}
		}
		zero_2d_arr(l->inputs, MAX_UNROLL_LENGTH, l->input_dimension);
		zero_2d_arr(l->input_gradients, MAX_UNROLL_LENGTH, l->input_dimension);
		l->t = 0;
		l = l->output_layer;
	}
	zero_2d_arr(n->cost_gradients, MAX_UNROLL_LENGTH, n->tail->size);
	n->t = 0;
}	

/*
 * Used to initialize & allocate memory for a layer of LSTM cells
 */
LSTM_layer *createLSTM_layer(size_t input_dim, size_t size, float plasticity){
	LSTM_layer *l = (LSTM_layer*)malloc(sizeof(LSTM_layer));
	l->plasticity = plasticity;
	l->input_dimension = input_dim + size;
	l->input_layer = NULL;
	l->output_layer = NULL;
	l->t = 0;

	Cell *cells = (Cell*)malloc(size*sizeof(Cell));
	for(long i = 0; i < size; i++){
		Cell *cell = &cells[i];
		
		//allocate weights for this cell
		float *input_nonl_weights = ALLOCATE(float, l->input_dimension);
		float *input_gate_weights = ALLOCATE(float, l->input_dimension);
		float *forget_gate_weights = ALLOCATE(float, l->input_dimension);
		float *output_gate_weights = ALLOCATE(float, l->input_dimension);

		//randomly initialize weights
		for(long j = 0; j < l->input_dimension; j++){
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
		cell->input_nonl = createGate(input_nonl_weights, input_nonl_bias, MAX_UNROLL_LENGTH, l->input_dimension);
		cell->input_gate = createGate(input_gate_weights, input_gate_bias, MAX_UNROLL_LENGTH, l->input_dimension);
		cell->forget_gate = createGate(forget_gate_weights, forget_gate_bias, MAX_UNROLL_LENGTH, l->input_dimension);
		cell->output_gate = createGate(output_gate_weights, output_gate_bias, MAX_UNROLL_LENGTH, l->input_dimension);

		cell->output = ALLOCATE(float, MAX_UNROLL_LENGTH);
		cell->state = ALLOCATE(float, MAX_UNROLL_LENGTH);
		cell->dstate = ALLOCATE(float, MAX_UNROLL_LENGTH);
		cell->gradient = ALLOCATE(float, MAX_UNROLL_LENGTH);
		cell->dOutput = ALLOCATE(float, MAX_UNROLL_LENGTH);
		cell->lstate = 0;
		cell->loutput = 0;

	}
	l->cells = cells;
	l->size = size;

	l->output = ALLOCATE(float, l->size);
	l->hidden = ALLOCATE(float, l->size);
	
	//allocate the 2d array used to store inputs over timesteps
	l->inputs = ALLOCATE(float*, MAX_UNROLL_LENGTH);
	for(long t = 0; t < MAX_UNROLL_LENGTH; t++) l->inputs[t] = ALLOCATE(float, l->input_dimension);
	
	//allocate the 2d array used to store gradients to be backpropagated over timesteps
	l->input_gradients = ALLOCATE(float*, MAX_UNROLL_LENGTH);
	for(long t = 0; t < MAX_UNROLL_LENGTH; t++) l->input_gradients[t] = ALLOCATE(float, l->input_dimension);

	zero_2d_arr(l->inputs, MAX_UNROLL_LENGTH, l->input_dimension);
	zero_2d_arr(l->input_gradients, MAX_UNROLL_LENGTH, l->input_dimension);

	return l;
}

/*
 * Called through a macro which allows a variable number of parameters
 */
LSTM lstm_from_arr(size_t *arr, size_t len){
	LSTM n;
	n.collapse = 0;
	n.stateful = 0;
	n.seq_len = 50;
	n.plasticity = 0.01;
	n.t = 0;
	n.head = NULL;
	n.tail = NULL;
	if(len < 3){
		printf("ERROR: lstm_from_arr(): must have at least input dim, hidden layer size, and output dim (3 layers), but only %d provided.\n", len);
		exit(1);
	}
	for(long i = 1; i < len-1; i++){

		LSTM_layer *l = createLSTM_layer(arr[i-1], arr[i], n.plasticity);

		if(n.head == NULL){ //initialize network
			n.head = l;
			n.tail = l;
		}else{
			n.tail->output_layer = l;
			l->input_layer = n.tail;
			n.tail = n.tail->output_layer;
		}
	}	

	//Allocate the 2d array to store the gradients calculated by the output layer
	n.cost_gradients = (float**)malloc(MAX_UNROLL_LENGTH * sizeof(float*));
	for(long i = 0; i < MAX_UNROLL_LENGTH; i++) n.cost_gradients[i] = (float*)malloc(arr[len-2] * sizeof(float));

	zero_2d_arr(n.cost_gradients, MAX_UNROLL_LENGTH, arr[len-2]);
//Will need to change once mlp.c is rewritten
/******************************/ 
	MLP output = initMLP();
	addLayer(&output, arr[len-2]);
	addLayer(&output, arr[len-1]);
	n.output_layer = output;
/******************************/

	return n;
}

float hypertan_element(float x){
	if(x > 7.0)  return 0.999998;
	if(x < -7.0) return -0.999998;
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
	for(long i = 0; i < length; i++){
		sum += x[i] * y[i];	
	}
	return sum;
}

/*
 * Computes the weight nudges to each parameter using backpropagation through time (bptt) for a single layer.
 */
void layer_backward(LSTM_layer *l, float **gradients, float plasticity){
	int recurrent_offset = l->input_dimension - l->size;
	//copy gradients over
	for(long t = l->t; t >= 0; t--){
		for(long j = 0; j < l->size; j++){
			l->cells[j].gradient[t] = gradients[t][j];
			float grad = l->cells[j].gradient[t];
			if(grad > 10000 || grad < -10000){
				printf("ERROR: layer_backward(): grad is unusually large (%6.5f) at t: %d, j: %d\n", grad, t, j);
				exit(1);
			}
			if(isnan(grad)){
				printf("ERROR: layer_backward(): gradient received from output layer was nan for cell %d, t %d!\n", j, t);
				exit(1);
			}
//			if(grad > 100) l->cells[j].gradient[t] = 100;
//			if(grad < -100)l->cells[j].gradient[t] = -100;
		}
	}
	//do the gradient calculations
//	zero_2d_arr(l->input_gradients, MAX_UNROLL_LENGTH, l->input_dimension);
	size_t MAX_TIME = l->t-1;
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
				delta_out = l->input_gradients[t+1][recurrent_offset + j];
				next_dstate = c->dstate[t+1];
				next_forget = f->output[t+1];
			}

			c->dOutput[t] = c->gradient[t] + delta_out;
			c->dstate[t] = c->dOutput[t] * o->output[t] * d_hypertan_element(c->state[t]) + next_dstate * next_forget;

			a->gradient[t] = c->dstate[t] * i->output[t] * a->dOutput[t];
			i->gradient[t] = c->dstate[t] * a->output[t] * i->dOutput[t];
			if(t) f->gradient[t] = c->dstate[t] * c->state[t-1] * f->dOutput[t];
			else  f->gradient[t] = 0;
			o->gradient[t] = c->dOutput[t] * hypertan_element(c->state[t]) * o->output[t] * (1 - o->output[t]);

			for(long k = 0; k < l->input_dimension; k++){
				l->input_gradients[t][k] += a->gradient[t] * a->weights[k];
				l->input_gradients[t][k] += i->gradient[t] * i->weights[k];
				l->input_gradients[t][k] += f->gradient[t] * f->weights[k];
				l->input_gradients[t][k] += o->gradient[t] * o->weights[k];
			}
		}
		//clip gradients that are too large (to put a band-aid on exploding gradients)
		for(long j = 0; j < l->input_dimension; j++){
			if(l->input_gradients[t][j] > MAX_GRAD) l->input_gradients[t][j] = MAX_GRAD;
			if(l->input_gradients[t][j] < -MAX_GRAD) l->input_gradients[t][j] = -MAX_GRAD;
		}
	}
	//do the parameter nudges
	for(long t = 0; t <= MAX_TIME; t++){
		for(long j = 0; j < l->size; j++){
			Cell *c = &l->cells[j];
			Gate *a = &c->input_nonl;
			Gate *i = &c->input_gate;
			Gate *f = &c->forget_gate;
			Gate *o = &c->output_gate;
			for(long k = 0; k < recurrent_offset; k++){
				a->weights[k] += a->gradient[t] * l->inputs[t][k] * plasticity;
				i->weights[k] += i->gradient[t] * l->inputs[t][k] * plasticity;
				f->weights[k] += f->gradient[t] * l->inputs[t][k] * plasticity;
				o->weights[k] += o->gradient[t] * l->inputs[t][k] * plasticity;
			}
			if(t < MAX_TIME){
				for(long k = recurrent_offset; k < l->input_dimension; k++){
					a->weights[k] += a->gradient[t+1] * c->output[t] * plasticity;
					i->weights[k] += i->gradient[t+1] * c->output[t] * plasticity;
					f->weights[k] += f->gradient[t+1] * c->output[t] * plasticity;
					o->weights[k] += o->gradient[t+1] * c->output[t] * plasticity;
					/*
					a->weights[k] += a->gradient[t] * c->output[t] * plasticity;
					i->weights[k] += i->gradient[t] * c->output[t] * plasticity;
					f->weights[k] += f->gradient[t] * c->output[t] * plasticity;
					o->weights[k] += o->gradient[t] * c->output[t] * plasticity;
					*/
					if(isnan(a->weights[k])){
						printf("ERROR: layer_backward(): nan'ed a weight while doing nudge, from %6.5f * %6.5f * %6.5f\n", a->gradient[t+1], c->output[t], plasticity);
						exit(1);
					}
				}
			}
			a->bias += a->gradient[t] * plasticity;
			i->bias += i->gradient[t] * plasticity;
			f->bias += f->gradient[t] * plasticity;
			o->bias += o->gradient[t] * plasticity;
		}
	}
}

/*
 * Computes the forward pass of a single layer
 */
void layer_forward(LSTM_layer *l, float *input){
	size_t t = l->t; //The current layer time
	size_t recurrent_offset = l->input_dimension - l->size;

	for(long i = 0; i < l->input_dimension - l->size; i++){
		l->inputs[t][i] = input[i]; //Copy input for this timestep 
	}
	for(long i = 0; i < l->size; i++){
		l->inputs[t][recurrent_offset + i] = l->cells[i].loutput; //Concatenate last timestep's outputs into our input vector
	}

	//Do output calculations for every cell in this layer
	for(long j = 0; j < l->size; j++){
		Cell *c = &l->cells[j];
		Gate *a = &c->input_nonl;
		Gate *i = &c->input_gate;
		Gate *f = &c->forget_gate;
		Gate *o = &c->output_gate;

		a->output[t] = hypertan_element(inner_product(a->weights, l->inputs[t], l->input_dimension) + a->bias); //input nonlinearity uses hypertangent
		i->output[t] = sigmoid_element(inner_product(i->weights, l->inputs[t], l->input_dimension) + i->bias); //all of the gates use sigmoid
		f->output[t] = sigmoid_element(inner_product(f->weights, l->inputs[t], l->input_dimension) + f->bias);
		o->output[t] = sigmoid_element(inner_product(o->weights, l->inputs[t], l->input_dimension) + o->bias);

		//not used, may delete
		a->dOutput[t] = 1 - a->output[t] * a->output[t];
		i->dOutput[t] = i->output[t] * (1 - i->output[t]);
		f->dOutput[t] = f->output[t] * (1 - f->output[t]);
		o->dOutput[t] = o->output[t] * (1 - o->output[t]);

		c->state[t] = a->output[t] * i->output[t] + f->output[t] * c->lstate; //Calculate the internal cell state
		c->output[t] = hypertan_element(c->state[t]) * o->output[t]; //Calculate the output of the cell

#if DEBUG
		if(isnan(a->output[t])){
			printf("ERROR: layer_forward(): c[%d], a->output[%d] is nan from tanh(%6.5f + %6.5f)\n", j, t, inner_product(a->weights, l->inputs[t], l->input_dimension), a->bias);
			printf("from inner product of:\n");
			for(int i = 0; i < l->input_dimension; i++){
				printf("%6.5f * %6.5f +\n", a->weights[i], l->inputs[t][i]);
			}
			exit(1);
		}
		if(isnan(c->state[t])){
			printf("ERROR: layer_forward(): nan while doing c[%d], state[%d] = %6.5f * %6.5f + %6.5f * %6.5f\n", j, t, a->output[t], i->output[t], f->output[t], c->lstate);
			exit(1);
		}
		if(isnan(c->output[t])){
			printf("ERROR: layer_forward(): c[%d]->output[%d] is nan from tanh(%6.5f * %6.5f)\n", j, t, c->state[t], o->output[t]);
			printf("                      : made %6.5f from %6.5f * %6.5f + %6.5f * %6.5f\n", c->state[t], a->output[t], i->output[t], f->output[t], c->lstate);
			exit(1);
		}
		if(c->state[t] > 1000 || c->state[t] < -1000){
			printf("WARNING: layer_forward(): c[%d]->state[%d] (%6.5f) is unusually large and may lead to exploding gradients!\n", j, t, c->state[t]);
			printf("                      : made %6.5f from %6.5f * %6.5f + %6.5f * %6.5f\n", c->state[t], a->output[t], i->output[t], f->output[t], c->lstate);
		}
#endif
		c->lstate = c->state[t]; //Set the last timestep's cell state to the current one for the next timestep
		c->loutput = c->output[t];
	}
	for(long i = 0; i < l->size; i++){
		l->output[i] = l->cells[i].output[t]; //copy cell outputs to the layer output vector (for use in inner_product)
		if(isnan(l->output[i])){
			printf("ERROR: layer_forward(): nan while copying to layer output vector: index %d, t %lu\n", i, t);
			exit(1);
		}
	}
}

/*
 * Calculates simple quadratic cost of network output.
 *
float quadratic_cost(LSTM *n, float *desired){
	size_t t = n->t;
	float cost = 0;
	for(long i = 0; i < n->tail->size; i++){
		n->cost_gradients[t][i] = desired[i] - n->tail->output[i];
#if SOFTMAX
		n->cost_gradients[t][i] *= n->tail->output[i] * (1 - n->tail->output[i]);
#endif
		cost += 0.5 * (desired[i] - n->tail->cells[i].output[t]) * (desired[i] - n->tail->cells[i].output[t]);

#if DEBUG
		if(isnan(n->cost_gradients[t][i])){
			printf("ERROR: quadratic_cost(): got a nan while doing cost gradients, %6.5f - %6.5f\n", desired[i], n->tail->output[i]);
			exit(1);
		}
		if(isnan(cost)){
			printf("ERROR: quadratic_cost(): got a nan while summing index i: %d, (%6.5f - %6.5f)^2\n", i, desired[i], n->tail->cells[i].output[t]);
			exit(1);
		}
		float grad = n->cost_gradients[t][i];
		if(grad > 1000 || grad < -1000){
			printf("ERROR: quadratic_cost(): unusually large gradient %6.5f on index %d from %6.5f - %6.5f\n", n->cost_gradients[t][i], i, desired[i], n->tail->output[i]);
			exit(1);
		}
#endif
	}
	return cost;
}

*
 * Calculates cross entropy cost of network output.
 
float cross_entropy_cost(LSTM *n, float *desired){
	size_t t = n->t;
	float cost = 0;
	for(long i = 0; i < n->tail->size; i++){
		float y_t = n->tail->output[i]; //output
		float l_t = desired[i]; //label
		if(y_t < 0.00001) y_t = 0.00001;
		if(y_t > 0.9999) y_t = .9999;


		n->cost_gradients[t][i] = l_t/y_t - (1 - l_t)/(1 - y_t); //cross entropy derivative
#if SOFTMAX
		n->cost_gradients[t][i] *= y_t * (1 - y_t); //softmax derivative
#endif

		float grad = n->cost_gradients[t][i];
		if(grad > 1000 || grad < -1000){
			printf("ERROR: cross_entropy_cost(): unusually large grad (%6.5f)\n", grad);
			exit(1);
		}
		cost += -(l_t * log(y_t) + (1 - l_t) * log(1 - y_t));
		if(isnan(cost)){
			printf("ERROR: cross_entropy_cost(): cost was NAN on output %d, t %lu, desired %6.5f and actual %6.5f\n", i, t, desired[i], n->tail->cells[i].output[t]);
			exit(0);
		}
		n->tail->output[i] = y_t;
	}
	return cost;
}
*/

/*
 * Performs a backward pass if the network unroll has been reached.
 */
float backward(LSTM *n, float *expected){
	MLP *output_mlp = &n->output_layer;
	float c = backpropagate(output_mlp, expected); //backpropagate the mlp output layer (not lstm layers)

	for(int i = 0; i < n->tail->size; i++){
		n->cost_gradients[n->t][i] = output_mlp->input->neurons[i].gradient; //copy gradients
	}

#if DEBUG
	if(n->seq_len > MAX_UNROLL_LENGTH){
		printf("ERROR: backward(): lstm.seq_len is too large (%d > max of %d).\n", n->seq_len, MAX_UNROLL_LENGTH);
		exit(1);
	}
	if(n->t > n->seq_len){
		printf("ERROR: backward(): lstm.t (%d) is greater than lstm.seq_len (%d) - something has gone wrong!\n", n->t, n->seq_len); 
		exit(1);
	}
#endif
	int weight_update = (n->t == n->seq_len-1) || n->collapse;
	float **grads = n->cost_gradients;
	LSTM_layer *l = n->tail;
	while(l){
		if(weight_update){
			layer_backward(l, grads, n->plasticity);
			l->t = 0;
			grads = l->input_gradients;
		}
		else l->t++;
		l = l->input_layer;
	}
	l = n->head;

	while(l && weight_update){
		zero_2d_arr(l->inputs, MAX_UNROLL_LENGTH, l->input_dimension);
		zero_2d_arr(l->input_gradients, MAX_UNROLL_LENGTH, l->input_dimension);
		l = l->output_layer;
	}
	if(weight_update){
		n->t = 0;
		n->collapse = 0;
		zero_2d_arr(n->cost_gradients, MAX_UNROLL_LENGTH, n->tail->size);
		//wipe(n);
	}
	else n->t++;
	return c;
}

/*
 * Does a forward pass through the network
 */
void forward(LSTM *n, float *x){
	/*
	printf("input: [");
	for(long i = 0; i < n->head->size; i++){
		printf("%6.5f", n->head->output[i]);
		if(i < n->head->size-1) printf(", ");
		else printf("]\n");
	}
	*/

	float *input = x;
	LSTM_layer *l = n->head;
	while(l){
		layer_forward(l, input);
		input = l->output;
		l = l->output_layer;
	}
	MLP *output_mlp = &n->output_layer;

	feedforward(output_mlp, n->tail->output);
	/*
	float sum = 0;
	printf("feedforward output: [");
	for(long i = 0; i < output_mlp->output->size; i++){
		printf("%6.5f", output_mlp->output->neurons[i].activation);
		sum += output_mlp->output->neurons[i].activation;
		if(i < output_mlp->output->size-1) printf(", ");
		else printf("], sum: %6.5f\n", sum);
	}
	*/
}

 /*
  * IO FUNCTIONS FOR READING AND WRITING TO A FILE
  */

static void writeToFile(FILE *fp, char *ptr){
  fprintf(fp, "%s", ptr);
  memset(ptr, '\0', strlen(ptr));
}

static void getWord(FILE *fp, char* dest){
  memset(dest, '\0', strlen(dest));
  int res = fscanf(fp, " %1023s", dest);
}

/*
 * Load the network's state from a file
 */
LSTM loadLSTMFromFile(const char *filename){
	FILE *fp = fopen(filename, "rb");
	char buff[1024];
	memset(buff, '\0', 1024);

	LSTM n;
	n.t = 0;
	n.collapse = 0;
	n.plasticity = 0.01;
	n.head = NULL;
	n.tail = NULL;

	//Because the lstm uses a mlp for its output layer, we need to load that mlp's params from a separate file (will rewrite when convenient)
	char *mlp_filename = (char*)malloc((strlen(filename)+5)*sizeof(char));
	strcpy(mlp_filename, filename);
	strcat(mlp_filename, ".mlp");
	n.output_layer = loadMLPFromFile(mlp_filename);
	free(mlp_filename);

	getWord(fp, buff);
	if(strcmp(buff, "LSTM") != 0){
		printf("ERROR: [%s] is not an LSTM.\n", buff);
		exit(1);
	}
	getWord(fp, buff); //read the number of layers from the .lstm file
	size_t size = strtol(buff, NULL, 10);

	getWord(fp, buff); //read the input dimension from the .lstm file
	size_t input_dimension = strtol(buff, NULL, 10);

	for(long i = 0; i < size; i++){
		getWord(fp, buff);
		if(strcmp(buff, "layer") == 0){
			getWord(fp, buff); //read the number of cells in this layer
			size_t layersize = strtol(buff, NULL, 10);
			LSTM_layer *l = createLSTM_layer(input_dimension, layersize, n.plasticity);

			for(long j = 0; j < layersize; j++){
				getWord(fp, buff);
				size_t numweights = strtol(buff, NULL, 10);
				for(long k = 0; k < numweights; k++){
					getWord(fp, buff);
					l->cells[j].input_nonl.weights[k] = strtod(buff, NULL);
					float weight = l->cells[j].input_nonl.weights[k];
					if(weight > 100 || weight < -100){
						printf("unusually large weight loaded (%6.5f)\n", weight);
					}
				}	
				getWord(fp, buff);
				l->cells[j].input_nonl.bias = strtod(buff, NULL);
				float bias = l->cells[j].input_nonl.bias;
				if(bias > 100 || bias < -100){
					printf("unusually large bias loaded (%6.5f)\n", bias);
				}
				for(long k = 0; k < numweights; k++){
					getWord(fp, buff);
					l->cells[j].input_gate.weights[k] = strtod(buff, NULL);
					float weight = l->cells[j].input_gate.weights[k];
					if(weight > 100 || weight < -100){
						printf("unusually large weight loaded (%6.5f)\n", weight);
					}
				}	
				getWord(fp, buff);
				l->cells[j].input_gate.bias = strtod(buff, NULL);
				bias = l->cells[j].input_gate.bias;
				if(bias > 100 || bias < -100){
					printf("unusually large bias loaded (%6.5f)\n", bias);
				}
				for(long k = 0; k < numweights; k++){
					getWord(fp, buff);
					l->cells[j].forget_gate.weights[k] = strtod(buff, NULL);
					float weight = l->cells[j].forget_gate.weights[k];
					if(weight > 100 || weight < -100){
						printf("unusually large weight loaded (%6.5f)\n", weight);
					}
				}	
				getWord(fp, buff);
				l->cells[j].forget_gate.bias = strtod(buff, NULL);
				bias = l->cells[j].forget_gate.bias;
				if(bias > 100 || bias < -100){
					printf("unusually large bias loaded (%6.5f)\n", bias);
				}
				for(long k = 0; k < numweights; k++){
					getWord(fp, buff);
					l->cells[j].output_gate.weights[k] = strtod(buff, NULL);
					float weight = l->cells[j].output_gate.weights[k];
					if(weight > 100 || weight < -100){
						printf("unusually large weight loaded (%6.5f)\n", weight);
					}
				}	
				getWord(fp, buff);
				l->cells[j].output_gate.bias = strtod(buff, NULL);
				bias = l->cells[j].output_gate.bias;
				if(bias > 100 || bias < -100){
					printf("unusually large bias loaded (%6.5f)\n", bias);
				}
			}
			if(!n.head){
				n.head = l;
				n.tail = l;
			}else{ 
				n.tail->output_layer = l;
				l->input_layer = n.tail;
				n.tail = l;
			}
			input_dimension = layersize;
		}
	}
	n.cost_gradients = (float**)malloc(MAX_UNROLL_LENGTH * sizeof(float*));
	for(long i = 0; i < MAX_UNROLL_LENGTH; i++) n.cost_gradients[i] = (float*)malloc(n.tail->size * sizeof(float));
	fclose(fp);
	wipe(&n);
	return n;
}

/*
 * Saves the network's state to a file that can be read later.
 */
void saveLSTMToFile(LSTM *n, char *filename){

	//Because the lstm uses a mlp for its output layer, we need to save that mlp's params to a separate file (will rewrite when convenient)
	char *mlp_filename = (char*)malloc((strlen(filename)+5)*sizeof(char));
	strcpy(mlp_filename, filename);
	strcat(mlp_filename, ".mlp");
	saveMLPToFile(&n->output_layer, mlp_filename);
	free(mlp_filename);

	FILE *fp;
	char buff[1024];
	memset(buff, '\0', 1024);

	//Create file
	fp = fopen(filename, "w");
	printf("Saving to: %s\n", filename);
	memset(buff, '\0', strlen(buff));

	size_t size = 0;
	LSTM_layer *current = n->head;
	while(current){
		size++;
		current = current->output_layer;
	}

	//Write header info to file
	strcat(buff, "LSTM ");
	writeToFile(fp, buff);
	snprintf(buff, 100, "%lu ", size);
	writeToFile(fp, buff);
	snprintf(buff, 100, "%lu ", n->head->input_dimension - n->head->size);
	writeToFile(fp, buff);

	current = n->head;
	for(long i = 0; i < size; i++){
		strcat(buff, "layer ");
		writeToFile(fp, buff);
		snprintf(buff, 100, "%lu ", current->size);
		writeToFile(fp, buff);
		for(long j = 0; j < current->size; j++){

			snprintf(buff, 100, "%lu ", current->input_dimension);
			writeToFile(fp, buff);
			for(long k = 0; k < current->input_dimension; k++){
				snprintf(buff, 100, "%f ", current->cells[j].input_nonl.weights[k]);
				writeToFile(fp, buff);
			}
			snprintf(buff, 100, "%f ", current->cells[j].input_nonl.bias);
			writeToFile(fp, buff);

			for(long k = 0; k < current->input_dimension; k++){
				snprintf(buff, 100, "%f ", current->cells[j].input_gate.weights[k]);
				writeToFile(fp, buff);
			}
			snprintf(buff, 100, "%f ", current->cells[j].input_gate.bias);
			writeToFile(fp, buff);

			for(long k = 0; k < current->input_dimension; k++){
				snprintf(buff, 100, "%f ", current->cells[j].forget_gate.weights[k]);
				writeToFile(fp, buff);
			}
			snprintf(buff, 100, "%f ", current->cells[j].forget_gate.bias);
			writeToFile(fp, buff);

			for(long k = 0; k < current->input_dimension; k++){
				snprintf(buff, 100, "%f ", current->cells[j].output_gate.weights[k]);
				writeToFile(fp, buff);
			}
			snprintf(buff, 100, "%f ", current->cells[j].output_gate.bias);
			writeToFile(fp, buff);
		}
		current = current->output_layer;
	}
	fclose(fp);
}
/*	
float step(LSTM_layer *l, float *input, float *desired){
	if(l->t >= MAX_UNROLL_LENGTH || desired == NULL){ //We've reached the max unroll length, so time to do bptt
		//bptt
		//printf("doing bptt\n");

		//do gradient math
		for(long t = l->t-1; t >= 0; t--){
//			printf("BACKPROP: doing timestep %d\n", t);
			for(long j = 0; j < l->size; j++){
				Cell *c = &l->cells[j];
				Gate *a = &c->input_nonl;
				Gate *i = &c->input_gate;
				Gate *f = &c->forget_gate;
				Gate *o = &c->output_gate;

				float delta_out;
				float next_dstate;
				float next_forget;

				if(t >= MAX_UNROLL_LENGTH-1){
					delta_out = 0; //Zero because no future timesteps
					next_dstate = 0;
					next_forget = 0;
				}else{
					delta_out = l->input_gradients[t+1][l->input_dimension - l->size + j];
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
				printf("BACKPROP t=%d: made d_state from %6.5f * %6.5f * %6.5f + %6.5f * %6.5f\n", t, c->dOutput[t], o->output[t], d_hypertan_element(c->state[t]), next_dstate, next_forget);
				printf("BACKPROP t=%d:		d_state: %6.5f\n", t, c->dstate[t]);
				printf("BACKPROP t=%d: 		d_a: %6.5f\n", t, a->gradient[t]);
				printf("BACKPROP t=%d:		d_i: %6.5f\n", t, i->gradient[t]);
				printf("BACKPROP t=%d:		d_f: %6.5f\n", t, f->gradient[t]);
				printf("BACKPROP t=%d:		d_o: %6.5f\n", t, o->gradient[t]);
				for(long k = 0; k < l->input_dimension; k++){
					l->input_gradients[t][k] += a->gradient[t] * a->weights[k];
					l->input_gradients[t][k] += i->gradient[t] * i->weights[k];
					l->input_gradients[t][k] += f->gradient[t] * f->weights[k];
					l->input_gradients[t][k] += o->gradient[t] * o->weights[k];
//					printf("BACKPROP t=%d: l->input_gradients[%d][%d] set to %f\n", t, t, k, l->input_gradients[t][k]);
				}
//				printf("BACKPROP t=%d: dOutput is %6.5f from gradient %6.5f and dout %6.5f\n", t, c->dOutput[t], c->gradient[t], delta_out); 
			}
		}
		//do the parameter nudges
		for(long t = 0; t < MAX_UNROLL_LENGTH; t++){
			for(long j = 0; j < l->size; j++){
				Cell *c = &l->cells[j];
				Gate *a = &c->input_nonl;
				Gate *i = &c->input_gate;
				Gate *f = &c->forget_gate;
				Gate *o = &c->output_gate;
				size_t recurrent_offset = l->input_dimension - l->size;
				for(long k = 0; k < recurrent_offset; k++){
//					printf("BACKPROP: adding %6.5f * %6.5f (%6.5f) to weight %d of a\n", a->gradient[t], l->inputs[t][k], a->gradient[t] * l->inputs[t][k], k);
					a->weights[k] += a->gradient[t] * l->inputs[t][k] * plasticity;
					i->weights[k] += i->gradient[t] * l->inputs[t][k] * plasticity;
					f->weights[k] += f->gradient[t] * l->inputs[t][k] * plasticity;
					o->weights[k] += o->gradient[t] * l->inputs[t][k] * plasticity;
				}
				if(t < MAX_UNROLL_LENGTH-1){
					for(long k = recurrent_offset; k < l->input_dimension; k++){
//						printf("BACKPROP: 		making a weight adjustment with a.grad[%d+1]: %6.5f * c.output[%d]: %6.5f\n", t, a->gradient[t+1], t, c->output[t]);
						a->weights[k] += a->gradient[t+1] * c->output[t] * plasticity;
						i->weights[k] += i->gradient[t+1] * c->output[t] * plasticity;
						f->weights[k] += f->gradient[t+1] * c->output[t] * plasticity;
						o->weights[k] += o->gradient[t+1] * c->output[t] * plasticity;
					}
				}
				a->bias += a->gradient[t] * plasticity;
				i->bias += i->gradient[t] * plasticity;
				f->bias += f->gradient[t] * plasticity;
				o->bias += o->gradient[t] * plasticity;
			}
		}
//		printf("weight 0: %6.5f\n", l->cells[0].input_gate.weights[0]);
//		printf("weight 1: %6.5f\n", l->cells[0].input_gate.weights[1]);
//		printf("weight 2: %6.5f\n", l->cells[0].input_gate.weights[2]);
//		printf("bias: %6.5f\n", l->cells[0].input_gate.bias);
		//end bptt
		l->t = 0;
		zero_2d_arr(l->inputs, MAX_UNROLL_LENGTH, l->input_dimension);
		zero_2d_arr(l->input_gradients, MAX_UNROLL_LENGTH, l->input_dimension);
	}
	size_t t = l->t;
	for(long i = 0; i < l->input_dimension - l->size; i++){
		l->inputs[t][i] = input[i];
	}
	for(long i = l->input_dimension - l->size; i < l->input_dimension; i++){

		if(t) l->inputs[t][i] = l->cells[i - (l->input_dimension - l->size)].output[t-1]; //recurrent input
		else  l->inputs[t][i] = l->cells[i - (l->input_dimension - l->size)].output[MAX_UNROLL_LENGTH-1];
	}

	//feedforward
	printf("FEEDFORWARD t=%lu: inputs: [", t);
	for(long j = 0; j < l->input_dimension; j++){
		printf("%2.3f", l->inputs[t][j]);
		if(j < l->input_dimension-1) printf(", ");
		else printf("]\n");
	}
	for(long j = 0; j < l->size; j++){
		Cell *c = &l->cells[j];
		Gate *a = &c->input_nonl;
		Gate *i = &c->input_gate;
		Gate *f = &c->forget_gate;
		Gate *o = &c->output_gate;

		a->output[t] = hypertan_element(inner_product(a->weights, l->inputs[t], l->input_dimension) + a->bias);
		i->output[t] = sigmoid_element(inner_product(i->weights, l->inputs[t], l->input_dimension) + i->bias);
		f->output[t] = sigmoid_element(inner_product(f->weights, l->inputs[t], l->input_dimension) + f->bias);
		o->output[t] = sigmoid_element(inner_product(o->weights, l->inputs[t], l->input_dimension) + o->bias);


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
	for(long j = 0; j < l->size; j++){
		l->cells[j].gradient[t] = desired[j] - l->cells[j].output[t];
//		printf("FEEDFORWARD t=%lu: Giving cell %d a gradient of %6.5f from %6.5f - %6.5f\n", t, j, l->cells[j].gradient[t], l->cells[j].output[t], desired[j]);
		cost += 0.5 * (desired[j] - l->cells[j].output[t]) * (desired[j] - l->cells[j].output[t]); //L2 loss
	}
	//end feedforward
	printf("FEEDFORWARD t=%lu: output: [", t);
	for(long j = 0; j < l->size; j++){
		printf("%2.3f", l->cells[j].output[t]);
		if(j < l->size-1) printf(", ");
		else printf("]\n");
	}

	l->t++;
	return cost;
}

*/
