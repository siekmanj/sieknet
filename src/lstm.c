/* Author: Jonah Siekmann
 * 10/3/2018
 * This is a simple implementation of a Long Short-Term Memory (LSTM) network, as described by Sepp Hochreiter and Jurgen Schmidhuber in their 1997 paper, with the addition of the forget gate described by Felix Gers.
 * Big thanks to Aidan Gomez for his wonderful numerical example of the backpropagation algorithm for an LSTM cell:
 * https://blog.aidangomez.ca/2016/04/17/Backpropogating-an-LSTM-A-Numerical-Example/
 */

#include "lstm.h"
#include "nonlinear.h"
#include <math.h>
#include <string.h>

#ifdef GPU
#include "opencl_utils.h"
#endif

#define ARR_FROM_GPU(name, gpumem, size) float name[size]; memset(name, '\0', size*sizeof(float)); check_error(clEnqueueReadBuffer(get_opencl_queue0(), gpumem, 1, 0, sizeof(float) * size, name, 0, NULL, NULL), "error reading from gpu (ARR_FROM_GPU)");

#define PRINTLIST(name, len) printf("printing %s: [", #name); for(int xyz = 0; xyz < len; xyz++){printf("%5.4f", name[xyz]); if(xyz < len-1) printf(", "); else printf("]\n");}
#define ALLOCATE(TYPE, NUM) (TYPE*)malloc((NUM) * (sizeof(TYPE)));

/*
 * Used to sample from softmax distribution, treats float
 * array as a normal distribution and returns a sample.
 */
int sample_softmax(float *probs, size_t len){
	float random_num = ((float)rand()) / (float)RAND_MAX;

	float lower_bound = 0;
  for(int i = 0; i < len; i++){
		float upper_bound = lower_bound + probs[i];
		if(random_num >= lower_bound && random_num < upper_bound){
			return i;
		}
		lower_bound += probs[i];
  }
  return len-1;
}

/*
 * Used as alternative to sampling from softmax distribution,
 * simply returns the largest number in a float array.
 */
int argmax(float *args, size_t len){
	int argm = 0;
	for(int i = 0; i < len; i++){
		if(args[argm] < args[i])
			argm = i;
	}
	return argm;
}

/*
 * Calculates the cost gradient for an lstm given a label vector y.
 * y is expected to be of size n.output_dimension. 
 */
float lstm_cost(LSTM *n, float *y){
	MLP_layer *mlp = &n->output_layer;
	//float tmp[n->output_dimension];
	float c = n->cost_fn(n->output, y, n->mlp_cost_gradient, n->output_dimension);

#ifndef GPU
	/* On the CPU, will run backward pass serially on softmax output layer, * 
	 * and copy resulting gradient into lstm's network_gradient[t] array.   */
	cpu_mlp_layer_backward(mlp, n->mlp_cost_gradient);
	float *grads = mlp->input_gradient;
#else
	/* On the GPU, will run backward pass in parallel, then use clEnqueueCopyBuffer *
	 * to copy resulting gradient into network_gradient[t].                         */

	//THIS MAY NOT WORK - TEST ON GPU
	printf("WARNING: NEED TO IMPLEMENT GPU-SIDE COST CALC HERE!!!\n");

	check_error(clEnqueueWriteBuffer(get_opencl_queue0(), n->mlp_cost_gradient, 0, 0, sizeof(float) * n->output_dimension, tmp, 0, NULL, NULL), "enqueuing cost gradient");
	gpu_mlp_layer_backward(mlp, n->mlp_cost_gradient, n->params, n->param_grad);
	cl_mem grads = mlp->input_gradient;
#endif
	
	size_t t = n->t;
	if(mlp->input_dimension != n->layers[n->depth-1].size){
		printf("ERROR: cost_wrapper(): Size mismatch between mlp output layer and final lstm layer (%lu vs %lu)\n", mlp->input_dimension, n->layers[n->depth-1].size);
		exit(1);
	}
#ifndef GPU
	/* CPU: copy gradient serially from mlp output layer to lstm network gradient. */
	for(int i = 0; i < mlp->input_dimension; i++)
		n->network_gradient[t][i] = grads[i];
#else
	/* GPU: copy gradient in parallel from mlp output layer to lstm network gradient */
	check_error(clEnqueueCopyBuffer(get_opencl_queue0(), grads, n->network_gradient[t], 0, 0, sizeof(float) * mlp->input_dimension, 0, NULL, NULL), "copying mlp grads to lstm network grads");
#endif

	/* increment network's timestep counter */
	n->t++;
	/* return the scalar cost from n->cost_fn. */
	return c;
}


#ifndef GPU

/********* BEGIN CPU-ONLY FUNCTIONS *******/

/*
 * Used to initialize input/forget/output gates
 */
static Gate createGate(float *weights, float *bias, float *weight_grad, float *bias_grad, size_t sequence_length){
	Gate g;
	g.output = ALLOCATE(float, sequence_length);
	g.gradient = ALLOCATE(float, sequence_length);

	g.weights = weights;
	g.bias = bias;

	g.bias_grad = bias_grad;
	g.weight_grad = weight_grad;

	return g;
}

/*
 * Used to reset the hidden state of the lstm
 */ 
static void cpu_wipe(LSTM *n){
	for(int i = 0; i < n->depth; i++){
		LSTM_layer *l = &n->layers[i];
		for(long t = 0; t < MAX_UNROLL_LENGTH; t++){
			for(long j = 0; j < l->size; j++){
				l->cells[j].state[t] = 0;
				l->cells[j].dstate[t] = 0;
				l->cells[j].lstate = 0;
				l->cells[j].loutput = 0;
			}
		}
		zero_2d_arr(l->output, MAX_UNROLL_LENGTH, l->size);
		zero_2d_arr(l->input_gradient, MAX_UNROLL_LENGTH, l->input_dimension);
	}
	zero_2d_arr(n->network_gradient, MAX_UNROLL_LENGTH, n->layers[n->depth-1].size);
	n->t = 0;
}	

/*
 * Used to initialize & allocate memory for a layer of LSTM cells
 */
LSTM_layer cpu_create_LSTM_layer(size_t input_dim, size_t size, float *param_addr, float *param_grad){
	LSTM_layer l;
	l.input_dimension = input_dim + size;

	Cell *cells = (Cell*)malloc(size*sizeof(Cell));

	int param_idx = 0;
	for(long i = 0; i < size; i++){
		Cell *cell = &cells[i];

		float *input_nonl_bias = &param_addr[param_idx];
		float *input_nonl_weights = &param_addr[param_idx+1];
		xavier_init(&param_addr[param_idx], (l.input_dimension+1), size);

		float *input_nonl_bias_grad = &param_grad[param_idx];
		float *input_nonl_weight_grad = &param_grad[param_idx+1];
		param_idx += l.input_dimension+1;

		float *input_gate_bias = &param_addr[param_idx];
		float *input_gate_weights = &param_addr[param_idx+1];
		xavier_init(&param_addr[param_idx], (l.input_dimension+1), size);

		float *input_gate_bias_grad = &param_grad[param_idx];
		float *input_gate_weight_grad = &param_grad[param_idx+1];
		param_idx += l.input_dimension+1;
		
		float *forget_gate_bias = &param_addr[param_idx];
		float *forget_gate_weights = &param_addr[param_idx+1];
		xavier_init(&param_addr[param_idx], (l.input_dimension+1), size);

		float *forget_gate_bias_grad = &param_grad[param_idx];
		float *forget_gate_weight_grad = &param_grad[param_idx+1];
		param_idx += l.input_dimension+1;
		
		float *output_gate_bias = &param_addr[param_idx];
		float *output_gate_weights = &param_addr[param_idx+1];
		xavier_init(&param_addr[param_idx], (l.input_dimension+1), size);

		float *output_gate_bias_grad = &param_grad[param_idx];
		float *output_gate_weight_grad = &param_grad[param_idx+1];
		param_idx += l.input_dimension+1;
		
		//Allocate gates
		cell->input_nonl = createGate(input_nonl_weights, input_nonl_bias, input_nonl_weight_grad, input_nonl_bias_grad, MAX_UNROLL_LENGTH);
		cell->input_gate = createGate(input_gate_weights, input_gate_bias, input_gate_weight_grad, input_gate_bias_grad, MAX_UNROLL_LENGTH);
		cell->forget_gate = createGate(forget_gate_weights, forget_gate_bias, forget_gate_weight_grad, forget_gate_bias_grad, MAX_UNROLL_LENGTH);
		cell->output_gate = createGate(output_gate_weights, output_gate_bias, output_gate_weight_grad,  output_gate_bias_grad, MAX_UNROLL_LENGTH);

		cell->state = ALLOCATE(float, MAX_UNROLL_LENGTH);
		cell->dstate = ALLOCATE(float, MAX_UNROLL_LENGTH);
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
 * Creates an LSTM for the CPU from an array of layer sizes. Called through a
 * macro create_lstm(...), which allows a variable number of parameters.
 */
LSTM cpu_lstm_from_arr(size_t *arr, size_t len){
	LSTM n;
	n.t = 0;
	n.stateful = 0;
	n.seq_len = 25;
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
	n.param_grad = ALLOCATE(float, num_params);

	memset(n.param_grad, '\0', num_params*sizeof(float));
	
	int param_idx = 0;
	for(int i = 1; i < len-1; i++){
		LSTM_layer l = cpu_create_LSTM_layer(arr[i-1], arr[i], &n.params[param_idx], &n.param_grad[param_idx]);
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

	n.mlp_cost_gradient = ALLOCATE(float, n.output_dimension);

	zero_2d_arr(n.network_gradient, MAX_UNROLL_LENGTH, arr[len-2]);
	zero_2d_arr(n.network_input, MAX_UNROLL_LENGTH, arr[0]);

	n.output_layer = cpu_create_MLP_layer(arr[len-2], arr[len-1], &n.params[param_idx], &n.param_grad[param_idx], softmax);

	//n.output_layer = cpu_create_MLP_layer(arr[len-2], arr[len-1], n.params, n.param_grad, softmax); //this should not work but does??
	
	n.output = n.output_layer.output;
	return n;
}

/*
 * Computes the forward pass of a single layer
 */
void cpu_lstm_layer_forward(LSTM_layer *l, float *input, size_t t){
	size_t recurrent_offset = l->input_dimension - l->size; /* offset of recurrent input in input array */

	l->input[t] = input; /* save pointer to input for backward pass */

	/* create a temporary input array, copy in last timestep's outputs as well as actual input array. */
	float tmp[l->input_dimension];
	for(int i = 0; i < recurrent_offset; i++)
		tmp[i] = input[i];
	for(int i = recurrent_offset; i < l->input_dimension; i++)
		tmp[i] = l->cells[i - recurrent_offset].loutput;

	/* do output calculations for every cell in this layer */
	for(long j = 0; j < l->size; j++){
		Cell *c = &l->cells[j];
		Gate *a = &c->input_nonl;
		Gate *i = &c->input_gate;
		Gate *f = &c->forget_gate;
		Gate *o = &c->output_gate;

		a->output[t] = activate(inner_product(a->weights, tmp, l->input_dimension) + *a->bias, hypertan); /* input nonlinearity uses hypertangent */
		i->output[t] = activate(inner_product(i->weights, tmp, l->input_dimension) + *i->bias, sigmoid);  /* all of the gates use sigmoid */
		f->output[t] = activate(inner_product(f->weights, tmp, l->input_dimension) + *f->bias, sigmoid);
		o->output[t] = activate(inner_product(o->weights, tmp, l->input_dimension) + *o->bias, sigmoid);

		c->state[t] = a->output[t] * i->output[t] + f->output[t] * c->lstate; //Calculate the internal cell state
		l->output[t][j] = HYPERTAN(c->state[t]) * o->output[t]; //Calculate the output of the cell

#ifdef DEBUG
		if(isnan(a->output[t])){
			printf("ERROR: lstm_layer_forward(): c[%ld], a->output[%lu] is nan from tanh(%6.5f + %6.5f)\n", j, t, inner_product(a->weights, l->input[t], l->input_dimension), *a->bias);
			printf("from inner product of:\n");
			for(int i = 0; i < l->input_dimension; i++){
				printf("%6.5f * %6.5f +\n", a->weights[i], l->input[t][i]);
			}
			exit(1);
		}
		if(isnan(c->state[t])){
			printf("ERROR: lstm_layer_forward(): nan while doing c[%ld], state[%lu] = %6.5f * %6.5f + %6.5f * %6.5f\n", j, t, a->output[t], i->output[t], f->output[t], c->lstate);
			exit(1);
		}
		if(isnan(l->output[t][j])){
			printf("ERROR: lstm_layer_forward(): c[%ld]->output[%lu] is nan from tanh(%6.5f * %6.5f)\n", j, t, c->state[t], o->output[t]);
			printf("                      : made %6.5f from %6.5f * %6.5f + %6.5f * %6.5f\n", c->state[t], a->output[t], i->output[t], f->output[t], c->lstate);
			exit(1);
		}
		if(c->state[t] > 8000 || c->state[t] < -8000){
			printf("WARNING: lstm_layer_forward(): c[%ld]->state[%lu] (%6.5f) is unusually large and may lead to exploding gradients!\n", j, t, c->state[t]);
			printf("                      : made %6.5f from %6.5f * %6.5f + %6.5f * %6.5f\n", c->state[t], a->output[t], i->output[t], f->output[t], c->lstate);
		}
#endif
		c->lstate = c->state[t]; //Set the last timestep's cell state to the current one for the next timestep
		c->loutput = l->output[t][j];

		if(c->lstate > SIEKNET_MAX_STATE) c->lstate = SIEKNET_MAX_STATE;
		if(c->lstate < -SIEKNET_MAX_STATE) c->lstate = -SIEKNET_MAX_STATE;
	}
}

/*
 * Does a forward pass through the network
 */
void cpu_lstm_forward(LSTM *n, float *x){
	size_t t = n->t;
	for(int i = 0; i < n->input_dimension; i++){
		n->network_input[t][i] = x[i];
	}
	//Feedforward through all LSTM layers
	float *input = n->network_input[t];
	for(int i = 0; i < n->depth; i++){
		LSTM_layer *l = &n->layers[i];
		cpu_lstm_layer_forward(l, input, t);
		input = l->output[t];
	}

	//Feedforward through final MLP layer
	cpu_mlp_layer_forward(&n->output_layer, input);
  if(!(rand()%3))
		n->guess = sample_softmax(n->output_layer.output, n->output_dimension);
	else
		n->guess = argmax(n->output_layer.output, n->output_dimension);
}


void cpu_lstm_layer_backward(LSTM_layer *l, float **grads, size_t MAX_TIME){
	int recurrent_offset = l->input_dimension - l->size;
	for(long t = MAX_TIME; t >= 0; t--){
		for(long j = 0; j < l->size; j++){
			Cell *c = &l->cells[j];
			Gate *a = &c->input_nonl;
			Gate *i = &c->input_gate;
			Gate *f = &c->forget_gate;
			Gate *o = &c->output_gate;

			float cell_grad;
			float next_dstate;
			float next_forget;

			if(t >= MAX_TIME){
				cell_grad = grads[t][j];
				next_dstate = 0;
				next_forget = 0;
			}else{
				cell_grad = grads[t][j] + l->input_gradient[t+1][recurrent_offset + j];
				next_dstate = c->dstate[t+1];
				next_forget = f->output[t+1];
			}
			c->dstate[t] = cell_grad * o->output[t] * D_HYPERTAN(HYPERTAN(c->state[t])) + next_dstate * next_forget;

			a->gradient[t] = c->dstate[t] * i->output[t] * differentiate(a->output[t], hypertan);
			i->gradient[t] = c->dstate[t] * a->output[t] * differentiate(i->output[t], sigmoid);
			if(t) f->gradient[t] = c->dstate[t] * c->state[t-1] * differentiate(f->output[t], sigmoid);
			else  f->gradient[t] = 0;
			o->gradient[t] = cell_grad * HYPERTAN(c->state[t]) * differentiate(o->output[t], sigmoid);

			for(long k = 0; k < l->input_dimension; k++){
				l->input_gradient[t][k] += a->gradient[t] * a->weights[k];
				l->input_gradient[t][k] += i->gradient[t] * i->weights[k];
				l->input_gradient[t][k] += f->gradient[t] * f->weights[k];
				l->input_gradient[t][k] += o->gradient[t] * o->weights[k];
			}
			for(long k = 0; k < recurrent_offset; k++){
				a->weight_grad[k] += a->gradient[t] * l->input[t][k];
				i->weight_grad[k] += i->gradient[t] * l->input[t][k];
				f->weight_grad[k] += f->gradient[t] * l->input[t][k];
				o->weight_grad[k] += o->gradient[t] * l->input[t][k];
			}
			if(t < MAX_TIME){
				for(long k = recurrent_offset; k < l->input_dimension; k++){
					a->weight_grad[k] += a->gradient[t+1] * l->output[t][j];
					i->weight_grad[k] += i->gradient[t+1] * l->output[t][j];
					f->weight_grad[k] += f->gradient[t+1] * l->output[t][j];
					o->weight_grad[k] += o->gradient[t+1] * l->output[t][j];
#ifdef STOP_ON_NAN
					if(isnan(a->weights[k])){
						printf("ERROR: layer_backward(): nan'ed a weight grad, from %6.5f * %6.5f\n", a->gradient[t+1], l->output[t][j]);
						exit(1);
					}
#endif
#ifdef SIEKNET_MAX_GRAD
					if(a->weight_grad[k] > SIEKNET_MAX_GRAD)       a->weight_grad[k] = SIEKNET_MAX_GRAD;
					else if(a->weight_grad[k] < -SIEKNET_MAX_GRAD) a->weight_grad[k] = -SIEKNET_MAX_GRAD;
					if(i->weight_grad[k] > SIEKNET_MAX_GRAD)       i->weight_grad[k] = SIEKNET_MAX_GRAD;
					else if(i->weight_grad[k] < -SIEKNET_MAX_GRAD) i->weight_grad[k] = -SIEKNET_MAX_GRAD;
					if(f->weight_grad[k] > SIEKNET_MAX_GRAD)       f->weight_grad[k] = SIEKNET_MAX_GRAD;
					else if(f->weight_grad[k] < -SIEKNET_MAX_GRAD) f->weight_grad[k] = -SIEKNET_MAX_GRAD;
					if(o->weight_grad[k] > SIEKNET_MAX_GRAD)       o->weight_grad[k] = SIEKNET_MAX_GRAD;
					else if(o->weight_grad[k] < -SIEKNET_MAX_GRAD) o->weight_grad[k] = -SIEKNET_MAX_GRAD;
#endif
				}
			}
			*a->bias_grad += a->gradient[t];
			*i->bias_grad += i->gradient[t];
			*f->bias_grad += f->gradient[t];
			*o->bias_grad += o->gradient[t];
#ifdef SIEKNET_MAX_GRAD
			if(*a->bias_grad > SIEKNET_MAX_GRAD)       *a->bias_grad = SIEKNET_MAX_GRAD;
			else if(*a->bias_grad < -SIEKNET_MAX_GRAD) *a->bias_grad = -SIEKNET_MAX_GRAD;
			if(*i->bias_grad > SIEKNET_MAX_GRAD)       *i->bias_grad = SIEKNET_MAX_GRAD;
			else if(*i->bias_grad < -SIEKNET_MAX_GRAD) *i->bias_grad = -SIEKNET_MAX_GRAD;
			if(*f->bias_grad > SIEKNET_MAX_GRAD)       *f->bias_grad = SIEKNET_MAX_GRAD;
			else if(*f->bias_grad < -SIEKNET_MAX_GRAD) *f->bias_grad = -SIEKNET_MAX_GRAD;
			if(*o->bias_grad > SIEKNET_MAX_GRAD)       *o->bias_grad = SIEKNET_MAX_GRAD;
			else if(*o->bias_grad < -SIEKNET_MAX_GRAD) *o->bias_grad = -SIEKNET_MAX_GRAD;
#endif
		}
	}
}

/*
 * Performs parameter update for all LSTM layers
 */
void cpu_lstm_backward(LSTM *n){
	if(n->t >= n->seq_len){
		float **grads = n->network_gradient;
		for(int i = n->depth-1; i >= 0; i--){
			cpu_lstm_layer_backward(&n->layers[i], grads, n->t-1);
			grads = n->layers[i].input_gradient;
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

/********  END CPU-ONLY FUNCTIONS  ********/

#else

/******** BEGIN GPU-ONLY FUNCTIONS ********/


cl_kernel make_onehot_kernel;
cl_kernel rnn_forward_kernel, rnn_backward_kernel;
cl_kernel lstm_forward_kernel;
cl_kernel lstm_input_gradient_kernel, lstm_parameter_gradient_kernel;
cl_kernel lstm_input_nonl_gradient_kernel, lstm_forget_gate_gradient_kernel, lstm_output_gate_gradient_kernel, lstm_dstate_kernel;
cl_kernel logistic_kernel, zero_init_kernel;

static int ARE_KERNELS_INITIALIZED = 0;
void lstm_kernel_setup(){
	mlp_kernel_setup();

	char *kernels[] = {"include/conf.h", "include/nonlinear.h", "src/lstm.kernel", "src/lstm.cl", "src/rnn.cl", "src/logistic.cl"};

	int err = 0;

	char *src = get_kernel_source(kernels, 6);
	cl_program prog = build_program(src);
	free(src);

	rnn_forward_kernel = clCreateKernel(prog, "rnn_forward_kernel", &err);
	check_error(err, "couldn't make recurrent forward kernel");

	lstm_dstate_kernel = clCreateKernel(prog, "lstm_dstate_kernel", &err);
	check_error(err, "couldn't make lstm dstate kernel");

	lstm_input_nonl_gradient_kernel = clCreateKernel(prog, "lstm_input_nonl_gradient_kernel", &err);
	check_error(err, "couldn't make lstm input gate grad kernel");

	lstm_forget_gate_gradient_kernel = clCreateKernel(prog, "lstm_forget_gate_gradient_kernel", &err);
	check_error(err, "couldn't make lstm forget gate grad kernel");

	lstm_output_gate_gradient_kernel = clCreateKernel(prog, "lstm_output_gate_gradient_kernel", &err);
	check_error(err, "couldn't make lstm output gate grad kernel");

	lstm_input_gradient_kernel = clCreateKernel(prog, "lstm_input_gradient_kernel", &err);
	check_error(err, "couldn't make lstm input grad kernel");

	lstm_parameter_gradient_kernel = clCreateKernel(prog, "lstm_parameter_gradient_kernel", &err);
	check_error(err, "couldn't make lstm parameter grad kernel");

	logistic_kernel = clCreateKernel(prog, "logistic_kernel", &err);
	check_error(err, "couldn't make logistic kernel");

	lstm_forward_kernel = clCreateKernel(prog, "lstm_forward_kernel", &err);
	check_error(err, "couldn't make linear forward kernel");

	zero_init_kernel = clCreateKernel(prog, "zero_init_kernel", &err);
	check_error(err, "couldn't make zero init kernel");

	make_onehot_kernel = clCreateKernel(prog, "make_onehot_kernel", &err);
	check_error(err, "couldn't make onehot kernel");


	
}

void gpu_zero_2d_arr(cl_mem *arr, size_t num, size_t len){
	for(int i = 0; i < num; i++){
		check_error(clSetKernelArg(zero_init_kernel, 0, sizeof(cl_mem), &arr[i]), "couldn't set zero kernel arg");
		check_error(clEnqueueNDRangeKernel(get_opencl_queue0(), zero_init_kernel, 1, NULL, &len, NULL, 0, NULL, NULL), "couldn't use zero kernel");
	}
}

void gpu_wipe(LSTM *n){
	gpu_zero_2d_arr(n->network_gradient, MAX_UNROLL_LENGTH, n->input_dimension);
	gpu_zero_2d_arr(n->network_input,    MAX_UNROLL_LENGTH, n->input_dimension);

	for(int i = 0; i < n->depth; i++){
		LSTM_layer *l = &n->layers[i];
		check_error(clSetKernelArg(zero_init_kernel, 0, sizeof(cl_mem), &l->loutput), "couldn't set zero kernel arg 0");
		check_error(clEnqueueNDRangeKernel(get_opencl_queue0(), zero_init_kernel, 1, NULL, &l->size, NULL, 0, NULL, NULL), "couldn't use zero kernel");
		check_error(clSetKernelArg(zero_init_kernel, 0, sizeof(cl_mem), &l->lstate), "couldn't set zero kernel arg 0");
		check_error(clEnqueueNDRangeKernel(get_opencl_queue0(), zero_init_kernel, 1, NULL, &l->size, NULL, 0, NULL, NULL), "couldn't use zero kernel");

		gpu_zero_2d_arr(l->output,             MAX_UNROLL_LENGTH, l->size);
		gpu_zero_2d_arr(l->cell_state,         MAX_UNROLL_LENGTH, l->size);
		gpu_zero_2d_arr(l->input_nonl_output,  MAX_UNROLL_LENGTH, l->size);
		gpu_zero_2d_arr(l->input_gate_output,  MAX_UNROLL_LENGTH, l->size);
		gpu_zero_2d_arr(l->forget_gate_output, MAX_UNROLL_LENGTH, l->size);
		gpu_zero_2d_arr(l->output_gate_output, MAX_UNROLL_LENGTH, l->size);
	}
}

LSTM_layer gpu_create_LSTM_layer(size_t input_dim, size_t size, cl_mem params, int param_offset){
	LSTM_layer l;
	l.size = size;
	l.param_offset = param_offset;
	l.input_dimension = input_dim + size;

	size_t neuron_offset = 0;
	float *tmp = (float*)malloc(sizeof(float)*l.size*(l.input_dimension+1)*4);
	for(int i = 0; i < size*4; i++){

		//xavier_init(&params[param_offset + neuron_offset], (l.input_dimension+1), size);
		xavier_init(&tmp[neuron_offset], (l.input_dimension+1), l.size);
		neuron_offset += l.input_dimension+1;
	}
	check_error(clEnqueueWriteBuffer(get_opencl_queue0(), params, 1, param_offset * sizeof(float), sizeof(float)*l.size*(l.input_dimension+1)*4, tmp, 0, NULL, NULL), "could not enqueue layer params");

	l.input_nonl_z  = ALLOCATE(cl_mem, MAX_UNROLL_LENGTH);
	l.input_gate_z  = ALLOCATE(cl_mem, MAX_UNROLL_LENGTH);
	l.forget_gate_z = ALLOCATE(cl_mem, MAX_UNROLL_LENGTH);
	l.output_gate_z = ALLOCATE(cl_mem, MAX_UNROLL_LENGTH);

	l.input_nonl_output  = ALLOCATE(cl_mem, MAX_UNROLL_LENGTH);
	l.input_gate_output  = ALLOCATE(cl_mem, MAX_UNROLL_LENGTH);
	l.forget_gate_output = ALLOCATE(cl_mem, MAX_UNROLL_LENGTH);
	l.output_gate_output = ALLOCATE(cl_mem, MAX_UNROLL_LENGTH);

	l.input_nonl_gradient  = ALLOCATE(cl_mem, MAX_UNROLL_LENGTH);
	l.input_gate_gradient  = ALLOCATE(cl_mem, MAX_UNROLL_LENGTH);
	l.forget_gate_gradient = ALLOCATE(cl_mem, MAX_UNROLL_LENGTH);
	l.output_gate_gradient = ALLOCATE(cl_mem, MAX_UNROLL_LENGTH);

	l.cell_state     = ALLOCATE(cl_mem, MAX_UNROLL_LENGTH);
	l.cell_dstate    = ALLOCATE(cl_mem, MAX_UNROLL_LENGTH);
	l.cell_gradient  = ALLOCATE(cl_mem, MAX_UNROLL_LENGTH);
	l.input_gradient = ALLOCATE(cl_mem, MAX_UNROLL_LENGTH);

	l.input  = ALLOCATE(cl_mem, MAX_UNROLL_LENGTH);
	l.output = ALLOCATE(cl_mem, MAX_UNROLL_LENGTH);

	int err;
	for(int t = 0; t < MAX_UNROLL_LENGTH; t++){
		l.input_nonl_z[t]  = clCreateBuffer(get_opencl_context(), CL_MEM_READ_WRITE, sizeof(float) * l.size, NULL, &err); 
		check_error(err, "allocating internal lstm memory");
		l.input_gate_z[t]  = clCreateBuffer(get_opencl_context(), CL_MEM_READ_WRITE, sizeof(float) * l.size, NULL, &err); 
		check_error(err, "allocating internal lstm memory");
		l.forget_gate_z[t] = clCreateBuffer(get_opencl_context(), CL_MEM_READ_WRITE, sizeof(float) * l.size, NULL, &err); 
		check_error(err, "allocating internal lstm memory");
		l.output_gate_z[t] = clCreateBuffer(get_opencl_context(), CL_MEM_READ_WRITE, sizeof(float) * l.size, NULL, &err); 
		check_error(err, "allocating internal lstm memory");

		l.input_nonl_output[t]  = clCreateBuffer(get_opencl_context(), CL_MEM_READ_WRITE, sizeof(float) * l.size, NULL, &err); 
		check_error(err, "allocating internal lstm memory");
		l.input_gate_output[t]  = clCreateBuffer(get_opencl_context(), CL_MEM_READ_WRITE, sizeof(float) * l.size, NULL, &err); 
		check_error(err, "allocating internal lstm memory");
		l.forget_gate_output[t] = clCreateBuffer(get_opencl_context(), CL_MEM_READ_WRITE, sizeof(float) * l.size, NULL, &err); 
		check_error(err, "allocating internal lstm memory");
		l.output_gate_output[t] = clCreateBuffer(get_opencl_context(), CL_MEM_READ_WRITE, sizeof(float) * l.size, NULL, &err); 
		check_error(err, "allocating internal lstm memory");

		l.input_nonl_gradient[t]  = clCreateBuffer(get_opencl_context(), CL_MEM_READ_WRITE, sizeof(float) * l.size, NULL, &err); 
		check_error(err, "allocating internal lstm memory");
		l.input_gate_gradient[t]  = clCreateBuffer(get_opencl_context(), CL_MEM_READ_WRITE, sizeof(float) * l.size, NULL, &err); 
		check_error(err, "allocating internal lstm memory");
		l.forget_gate_gradient[t] = clCreateBuffer(get_opencl_context(), CL_MEM_READ_WRITE, sizeof(float) * l.size, NULL, &err); 
		check_error(err, "allocating internal lstm memory");
		l.output_gate_gradient[t] = clCreateBuffer(get_opencl_context(), CL_MEM_READ_WRITE, sizeof(float) * l.size, NULL, &err); 
		check_error(err, "allocating internal lstm memory");

		l.output[t] = clCreateBuffer(get_opencl_context(), CL_MEM_READ_WRITE, sizeof(float) * l.size, NULL, &err); 
		check_error(err, "allocating internal lstm memory");
		l.cell_state[t] = clCreateBuffer(get_opencl_context(), CL_MEM_READ_WRITE, sizeof(float) * l.size, NULL, &err); 
		check_error(err, "allocating internal lstm memory");

		l.cell_dstate[t] = clCreateBuffer(get_opencl_context(), CL_MEM_READ_WRITE, sizeof(float) * l.size, NULL, &err);
		check_error(err, "allocating internal lstm memory");

		l.cell_gradient[t] = clCreateBuffer(get_opencl_context(), CL_MEM_READ_WRITE, sizeof(float) * l.size, NULL, &err);
		check_error(err, "allocating internal lstm memory");

		l.input_gradient[t] = clCreateBuffer(get_opencl_context(), CL_MEM_READ_WRITE, sizeof(float) * l.input_dimension, NULL, &err);
		check_error(err, "allocating internal lstm memory");
	}
	l.loutput = clCreateBuffer(get_opencl_context(), CL_MEM_READ_WRITE, sizeof(float) * l.size, NULL, &err); 
	l.lstate = clCreateBuffer(get_opencl_context(), CL_MEM_READ_WRITE, sizeof(float) * l.size, NULL, &err); 
	check_error(err, "allocating internal lstm memory");

	return l;
}

LSTM gpu_lstm_from_arr(size_t *arr, size_t len){
	initialize_opencl();
	if(!ARE_KERNELS_INITIALIZED)
		lstm_kernel_setup();

	LSTM n;
	n.t = 0;
	n.stateful = 0;
	n.seq_len = 25;
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
	
	int err;
	n.params = clCreateBuffer(get_opencl_context(), CL_MEM_READ_WRITE, sizeof(float) * n.num_params, NULL, &err);
	check_error(err, "creating gpu params");
	n.param_grad = clCreateBuffer(get_opencl_context(), CL_MEM_READ_WRITE, sizeof(float) * n.num_params, NULL, &err);
	check_error(err, "creating param grad");

	n.network_gradient = ALLOCATE(cl_mem, MAX_UNROLL_LENGTH);
	n.network_input    = ALLOCATE(cl_mem, MAX_UNROLL_LENGTH);
	n.mlp_cost_gradient= clCreateBuffer(get_opencl_context(), CL_MEM_READ_WRITE, sizeof(float) * arr[len-1], NULL, &err);

	for(int t = 0; t < MAX_UNROLL_LENGTH; t++){
		n.network_gradient[t] = clCreateBuffer(get_opencl_context(), CL_MEM_READ_WRITE, sizeof(float) * arr[len-2], NULL, &err);
		check_error(err, "couldn't make internal lstm memory (network gradient)");
		n.network_input[t] = clCreateBuffer(get_opencl_context(), CL_MEM_READ_WRITE, sizeof(float) * arr[0], NULL, &err);
		check_error(err, "couldn't make internal lstm memory (network input)");
	}

	gpu_zero_2d_arr(n.network_gradient, MAX_UNROLL_LENGTH, arr[len-2]);
	gpu_zero_2d_arr(n.network_input, MAX_UNROLL_LENGTH, arr[0]);
	gpu_zero_2d_arr(&n.param_grad, 1, n.num_params);
	
	int param_idx = 0;
	for(int i = 1; i < len-1; i++){
		LSTM_layer l = gpu_create_LSTM_layer(arr[i-1], arr[i], n.params, param_idx);
		param_idx += (4*(arr[i-1]+arr[i]+1))*arr[i];
		n.layers[i-1] = l;
	}
	n.output_layer = gpu_create_MLP_layer(arr[len-2], arr[len-1], n.params, param_idx, softmax);
	n.output = ALLOCATE(float, n.output_dimension);

	gpu_wipe(&n);
	return n;
}

/* oh boy here we go */
static void gpu_lstm_layer_forward(LSTM_layer *l, cl_mem x, cl_mem params, size_t t, size_t num_p){
	l->input[t] = x;

	Nonlinearity gate_fn = sigmoid;
	Nonlinearity nonl_fn = hypertan;

	int params_per_gate  = (l->input_dimension+1);
	int input_nonl_base  = l->param_offset + 0 * params_per_gate;
	int input_gate_base  = l->param_offset + 1 * params_per_gate;
	int forget_gate_base = l->param_offset + 2 * params_per_gate;
	int output_gate_base = l->param_offset + 3 * params_per_gate;
	int skipdist         =                   4 * params_per_gate;

	check_error(clSetKernelArg(rnn_forward_kernel, 0, sizeof(cl_mem), &x), "setting forward kernel arg0");
	check_error(clSetKernelArg(rnn_forward_kernel, 1, sizeof(cl_mem), &l->loutput), "setting forward kernel arg1");
	check_error(clSetKernelArg(rnn_forward_kernel, 2, sizeof(cl_mem), &l->input_nonl_z[t]), "setting forward kernel arg2");
	check_error(clSetKernelArg(rnn_forward_kernel, 3, sizeof(cl_mem), &params), "setting forward kernel arg3");
	check_error(clSetKernelArg(rnn_forward_kernel, 4, sizeof(int), &l->input_dimension), "setting forward kernel arg4");
	check_error(clSetKernelArg(rnn_forward_kernel, 5, sizeof(int), &l->size), "setting forward kernel arg4");
	check_error(clSetKernelArg(rnn_forward_kernel, 6, sizeof(int), &input_nonl_base), "setting forward kernel arg5");
	check_error(clSetKernelArg(rnn_forward_kernel, 7, sizeof(int), &skipdist), "setting forward kernel arg6");
	check_error(clEnqueueNDRangeKernel(get_opencl_queue0(), rnn_forward_kernel, 1, NULL, &l->size, NULL, 0, NULL, NULL), "gpu_lstm_layer_forward(): couldn't enqueue linear kernel");
	
	check_error(clSetKernelArg(rnn_forward_kernel, 0, sizeof(cl_mem), &x), "setting forward kernel arg0");
	check_error(clSetKernelArg(rnn_forward_kernel, 1, sizeof(cl_mem), &l->loutput), "setting forward kernel arg1");
	check_error(clSetKernelArg(rnn_forward_kernel, 2, sizeof(cl_mem), &l->input_gate_z[t]), "setting forward kernel arg2");
	check_error(clSetKernelArg(rnn_forward_kernel, 3, sizeof(cl_mem), &params), "setting forward kernel arg3");
	check_error(clSetKernelArg(rnn_forward_kernel, 4, sizeof(int), &l->input_dimension), "setting forward kernel arg4");
	check_error(clSetKernelArg(rnn_forward_kernel, 5, sizeof(int), &l->size), "setting forward kernel arg4");
	check_error(clSetKernelArg(rnn_forward_kernel, 6, sizeof(int), &input_gate_base), "setting forward kernel arg5");
	check_error(clSetKernelArg(rnn_forward_kernel, 7, sizeof(int), &skipdist), "setting forward kernel arg6");
	check_error(clEnqueueNDRangeKernel(get_opencl_queue0(), rnn_forward_kernel, 1, NULL, &l->size, NULL, 0, NULL, NULL), "gpu_lstm_layer_forward(): couldn't enqueue linear recurrent kernel");

	check_error(clSetKernelArg(rnn_forward_kernel, 0, sizeof(cl_mem), &x), "setting forward kernel arg0");
	check_error(clSetKernelArg(rnn_forward_kernel, 1, sizeof(cl_mem), &l->loutput), "setting forward kernel arg1");
	check_error(clSetKernelArg(rnn_forward_kernel, 2, sizeof(cl_mem), &l->forget_gate_z[t]), "setting forward kernel arg2");
	check_error(clSetKernelArg(rnn_forward_kernel, 3, sizeof(cl_mem), &params), "setting forward kernel arg3");
	check_error(clSetKernelArg(rnn_forward_kernel, 4, sizeof(int), &l->input_dimension), "setting forward kernel arg4");
	check_error(clSetKernelArg(rnn_forward_kernel, 5, sizeof(int), &l->size), "setting forward kernel arg4");
	check_error(clSetKernelArg(rnn_forward_kernel, 6, sizeof(int), &forget_gate_base), "setting forward kernel arg5");
	check_error(clSetKernelArg(rnn_forward_kernel, 7, sizeof(int), &skipdist), "setting forward kernel arg6");
	check_error(clEnqueueNDRangeKernel(get_opencl_queue0(), rnn_forward_kernel, 1, NULL, &l->size, NULL, 0, NULL, NULL), "gpu_lstm_layer_forward(): couldn't enqueue linear recurrent kernel");

	check_error(clSetKernelArg(rnn_forward_kernel, 0, sizeof(cl_mem), &x), "setting forward kernel arg0");
	check_error(clSetKernelArg(rnn_forward_kernel, 1, sizeof(cl_mem), &l->loutput), "setting forward kernel arg1");
	check_error(clSetKernelArg(rnn_forward_kernel, 2, sizeof(cl_mem), &l->output_gate_z[t]), "setting forward kernel arg1");
	check_error(clSetKernelArg(rnn_forward_kernel, 3, sizeof(cl_mem), &params), "setting forward kernel arg2");
	check_error(clSetKernelArg(rnn_forward_kernel, 4, sizeof(int), &l->input_dimension), "setting forward kernel arg4");
	check_error(clSetKernelArg(rnn_forward_kernel, 5, sizeof(int), &l->size), "setting forward kernel arg4");
	check_error(clSetKernelArg(rnn_forward_kernel, 6, sizeof(int), &output_gate_base), "setting forward kernel arg5");
	check_error(clSetKernelArg(rnn_forward_kernel, 7, sizeof(int), &skipdist), "setting forward kernel arg6");
	check_error(clEnqueueNDRangeKernel(get_opencl_queue0(), rnn_forward_kernel, 1, NULL, &l->size, NULL, 0, NULL, NULL), "gpu_lstm_layer_forward(): couldn't enqueue linear recurrent kernel");
	
	check_error(clSetKernelArg(logistic_kernel, 0, sizeof(cl_mem), &l->input_nonl_z[t]), "setting logistic arg 0");
	check_error(clSetKernelArg(logistic_kernel, 1, sizeof(cl_mem), &l->input_nonl_output[t]), "setting logistic arg 0");
	check_error(clSetKernelArg(logistic_kernel, 2, sizeof(Nonlinearity), &nonl_fn), "setting logistic arg 0");
	//check_error(clFinish(get_opencl_queue0()), "waiting for queue 0 to finish executing (forward pass)");
	check_error(clEnqueueNDRangeKernel(get_opencl_queue0(), logistic_kernel, 1, NULL, &l->size, NULL, 0, NULL, NULL), "gpu_lstm_layer_forward(): couldn't enqueue logistic kernel");

	check_error(clSetKernelArg(logistic_kernel, 0, sizeof(cl_mem), &l->input_gate_z[t]), "setting logistic arg 0");
	check_error(clSetKernelArg(logistic_kernel, 1, sizeof(cl_mem), &l->input_gate_output[t]), "setting logistic arg 0");
	check_error(clSetKernelArg(logistic_kernel, 2, sizeof(Nonlinearity), &gate_fn), "setting logistic arg 0");
	//check_error(clFinish(get_opencl_queue1()), "waiting for queue 1 to finish executing (forward pass)");
	check_error(clEnqueueNDRangeKernel(get_opencl_queue0(), logistic_kernel, 1, NULL, &l->size, NULL, 0, NULL, NULL), "gpu_lstm_layer_forward(): couldn't enqueue logistic kernel");

	check_error(clSetKernelArg(logistic_kernel, 0, sizeof(cl_mem), &l->forget_gate_z[t]), "setting logistic arg 0");
	check_error(clSetKernelArg(logistic_kernel, 1, sizeof(cl_mem), &l->forget_gate_output[t]), "setting logistic arg 0");
	check_error(clSetKernelArg(logistic_kernel, 2, sizeof(Nonlinearity), &gate_fn), "setting logistic arg 0");
	//check_error(clFinish(get_opencl_queue2()), "waiting for queue 2 to finish executing (forward pass)");
	check_error(clEnqueueNDRangeKernel(get_opencl_queue0(), logistic_kernel, 1, NULL, &l->size, NULL, 0, NULL, NULL), "couldn't enqueue logistic kernel");

	check_error(clSetKernelArg(logistic_kernel, 0, sizeof(cl_mem), &l->output_gate_z[t]), "setting logistic arg 0");
	check_error(clSetKernelArg(logistic_kernel, 1, sizeof(cl_mem), &l->output_gate_output[t]), "setting logistic arg 1");
	check_error(clSetKernelArg(logistic_kernel, 2, sizeof(Nonlinearity), &gate_fn), "setting logistic arg 2");
	//check_error(clFinish(get_opencl_queue3()), "waiting for queue 3 to finish executing (forward pass)");
	check_error(clEnqueueNDRangeKernel(get_opencl_queue0(), logistic_kernel, 1, NULL, &l->size, NULL, 0, NULL, NULL), "couldn't enqueue logistic kernel");

	check_error(clSetKernelArg(lstm_forward_kernel, 0, sizeof(cl_mem), &l->input_nonl_output[t]), "setting lstm forward arg 0");
	check_error(clSetKernelArg(lstm_forward_kernel, 1, sizeof(cl_mem), &l->input_gate_output[t]), "setting lstm forward arg 1");
	check_error(clSetKernelArg(lstm_forward_kernel, 2, sizeof(cl_mem), &l->forget_gate_output[t]), "setting lstm forward arg 2");
	check_error(clSetKernelArg(lstm_forward_kernel, 3, sizeof(cl_mem), &l->output_gate_output[t]), "setting lstm forward arg 3");
	check_error(clSetKernelArg(lstm_forward_kernel, 4, sizeof(cl_mem), &l->cell_state[t]), "setting lstm forward arg 4");
	check_error(clSetKernelArg(lstm_forward_kernel, 5, sizeof(cl_mem), &l->lstate), "setting lstm forward arg 4");
	check_error(clSetKernelArg(lstm_forward_kernel, 6, sizeof(cl_mem), &l->output[t]), "setting lstm forward arg 4");
	//check_error(clFinish(get_opencl_queue0()), "waiting for queue 0 to finish executing (forward pass)");
	//check_error(clFinish(get_opencl_queue1()), "waiting for queue 1 to finish executing (forward pass)");
	//check_error(clFinish(get_opencl_queue2()), "waiting for queue 2 to finish executing (forward pass)");
	//check_error(clFinish(get_opencl_queue3()), "waiting for queue 3 to finish executing (forward pass)");
	check_error(clEnqueueNDRangeKernel(get_opencl_queue0(), lstm_forward_kernel, 1, NULL, &l->size, NULL, 0, NULL, NULL), "couldn't lstm forward kernel");

	check_error(clEnqueueCopyBuffer(get_opencl_queue0(), l->output[t], l->loutput, 0, 0, l->size * sizeof(float), 0, NULL, NULL), "copying output to loutput");
	check_error(clEnqueueCopyBuffer(get_opencl_queue0(), l->cell_state[t], l->lstate, 0, 0, l->size * sizeof(float), 0, NULL, NULL), "copying cell state to lcell_state");
	//check_error(clFinish(get_opencl_queue0()), "waiting for queue 0 to finish executing (forward pass)");
}

static void gpu_lstm_forward(LSTM *n, float *x){

	size_t t = n->t;
#ifdef SIEKNET_ONEHOT_SPEEDUP
	int m = argmax(x, n->input_dimension);
	check_error(clSetKernelArg(make_onehot_kernel, 0, sizeof(cl_mem), &n->network_input[t]), "couldn't set onehot arg 0");
	check_error(clSetKernelArg(make_onehot_kernel, 1, sizeof(int), &m), "couldn't set onehot arg 1");
	check_error(clEnqueueNDRangeKernel(get_opencl_queue0(), make_onehot_kernel, 1, NULL, &n->input_dimension, NULL, 0, NULL, NULL), "couldn't do onehot kernel");

#else
	check_error(clEnqueueWriteBuffer(get_opencl_queue0(), n->network_input[t], 1, 0, sizeof(float) * n->input_dimension, x, 0, NULL, NULL), "copying input");
#endif

	//Feedforward through all LSTM layers
	cl_mem input = n->network_input[t];
	for(int i = 0; i < n->depth; i++){
		LSTM_layer *l = &n->layers[i];
		gpu_lstm_layer_forward(l, input, n->params, t, n->num_params);
		input = l->output[t];
	}

	gpu_mlp_layer_forward(&n->output_layer, input, n->params);

	check_error(clEnqueueReadBuffer(get_opencl_queue0(), n->output_layer.output, 1, 0, sizeof(float) * n->output_layer.size, n->output, 0, NULL, NULL), "error reading output from gpu");

	check_error(clFinish(get_opencl_queue0()), "waiting for kernels to finish executing (forward pass)");

  if(!(rand()%3))
		n->guess = sample_softmax(n->output, n->output_dimension);
	else
		n->guess = argmax(n->output, n->output_dimension);

}

void simulate_ogate(float *gradient,
									 float *state,
									 float *output_gate_out,
									 float *future_input_gradient,
									 float *output_gate_gradient,
									 Nonlinearity gate_fn,
									 int recurrent_offset,
									 int use_future_grads,
									 int dim){
	for(int i = 0; i < dim; i++){
		float cell_grad;
		if(use_future_grads)
			cell_grad = gradient[i] + future_input_gradient[recurrent_offset + i];
		else
			cell_grad = gradient[i];

		output_gate_gradient[i] = cell_grad * HYPERTAN(state[i]) * differentiate(output_gate_out[i], gate_fn);
	}
}
static void gpu_lstm_layer_backward(LSTM_layer *l, cl_mem *grad, cl_mem params, cl_mem param_grad, size_t MAX_TIME){
	
	int recurrent_offset = l->input_dimension - l->size;
	
	Nonlinearity gate_fn = sigmoid;
	Nonlinearity nonl_fn = hypertan;

	int params_per_cell = (l->input_dimension+1)*4;

	for(long t = MAX_TIME; t >= 0; t--){
		int use_future_grads, use_past_outputs;
		if(t >= MAX_TIME) use_future_grads = 0;
		else              use_future_grads = 1;

		if(!t) use_past_outputs = 0;
		else   use_past_outputs = 1;

		/* calculate dstate */
		check_error(clSetKernelArg(lstm_dstate_kernel, 0, sizeof(cl_mem), &grad[t]), "lstm dstate kernel arg 0");
		check_error(clSetKernelArg(lstm_dstate_kernel, 1, sizeof(cl_mem), &l->cell_state[t]), "lstm dstate kernel arg 1");
		check_error(clSetKernelArg(lstm_dstate_kernel, 2, sizeof(cl_mem), &l->output_gate_output[t]), "lstm dstate kernel arg 2");
		if(use_future_grads){
			check_error(clSetKernelArg(lstm_dstate_kernel, 3, sizeof(cl_mem), &l->cell_dstate[t+1]), "lstm dstate kernel arg 3");
			check_error(clSetKernelArg(lstm_dstate_kernel, 4, sizeof(cl_mem), &l->forget_gate_output[t+1]), "lstm dstate kernel arg 4");
			check_error(clSetKernelArg(lstm_dstate_kernel, 5, sizeof(cl_mem), &l->input_gradient[t+1]), "lstm dstate kernel arg 5");
		}else{
			check_error(clSetKernelArg(lstm_dstate_kernel, 3, sizeof(cl_mem), &l->cell_dstate[t]), "lstm dstate kernel arg 3");
			check_error(clSetKernelArg(lstm_dstate_kernel, 4, sizeof(cl_mem), &l->forget_gate_output[t]), "lstm dstate kernel arg 4");
			check_error(clSetKernelArg(lstm_dstate_kernel, 5, sizeof(cl_mem), &l->input_gradient[t]), "lstm dstate kernel arg 5");
		}
		check_error(clSetKernelArg(lstm_dstate_kernel, 6, sizeof(cl_mem), &l->cell_dstate[t]), "lstm dstate kernel arg 7");
		check_error(clSetKernelArg(lstm_dstate_kernel, 7, sizeof(int), &recurrent_offset), "lstm dstate kernel arg 8");
		check_error(clSetKernelArg(lstm_dstate_kernel, 8, sizeof(int), &use_future_grads), "lstm dstate kernel arg 9");
		check_error(clEnqueueNDRangeKernel(get_opencl_queue0(), lstm_dstate_kernel, 1, NULL, &l->size, NULL, 0, NULL, NULL), "couldn't use dstate kernel");

		/* calculate input nonlinearity gradient */
		check_error(clSetKernelArg(lstm_input_nonl_gradient_kernel, 0, sizeof(cl_mem), &l->cell_dstate[t]), "lstm nonl grad kernel arg 0");
		check_error(clSetKernelArg(lstm_input_nonl_gradient_kernel, 1, sizeof(cl_mem), &l->input_gate_output[t]), "lstm nonl grad kernel arg 1");
		check_error(clSetKernelArg(lstm_input_nonl_gradient_kernel, 2, sizeof(cl_mem), &l->input_nonl_output[t]), "lstm nonl gradkernel arg 2");
		check_error(clSetKernelArg(lstm_input_nonl_gradient_kernel, 3, sizeof(cl_mem), &l->input_nonl_gradient[t]), "lstm nonl grad kernel arg 3");
		check_error(clSetKernelArg(lstm_input_nonl_gradient_kernel, 4, sizeof(Nonlinearity), &nonl_fn), "lstm nonl grad kernel arg 4");
		check_error(clEnqueueNDRangeKernel(get_opencl_queue0(), lstm_input_nonl_gradient_kernel, 1, NULL, &l->size, NULL, 0, NULL, NULL), "couldn't use dstate kernel");

		/* calculate input gate gradient (reuses input nonlinearity kernel) */
		check_error(clSetKernelArg(lstm_input_nonl_gradient_kernel, 0, sizeof(cl_mem), &l->cell_dstate[t]), "lstm gate grad kernel arg 0");
		check_error(clSetKernelArg(lstm_input_nonl_gradient_kernel, 1, sizeof(cl_mem), &l->input_nonl_output[t]), "lstm gate grad kernel arg 0");
		check_error(clSetKernelArg(lstm_input_nonl_gradient_kernel, 2, sizeof(cl_mem), &l->input_gate_output[t]), "lstm gate grad kernel arg 0");
		check_error(clSetKernelArg(lstm_input_nonl_gradient_kernel, 3, sizeof(cl_mem), &l->input_gate_gradient[t]), "lstm gate grad kernel arg 0");
		check_error(clSetKernelArg(lstm_input_nonl_gradient_kernel, 4, sizeof(Nonlinearity), &gate_fn), "lstm gate grad kernel arg 0");
		check_error(clEnqueueNDRangeKernel(get_opencl_queue0(), lstm_input_nonl_gradient_kernel, 1, NULL, &l->size, NULL, 0, NULL, NULL), "couldn't use dstate kernel");

		/* calculate forget gate gradient */
		check_error(clSetKernelArg(lstm_forget_gate_gradient_kernel, 0, sizeof(cl_mem), &l->cell_dstate[t]), "lstm fgate grad kernel arg 0");
		if(use_past_outputs)
			check_error(clSetKernelArg(lstm_forget_gate_gradient_kernel, 1, sizeof(cl_mem), &l->cell_state[t-1]), "lstm fgate grad kernel arg 1");
		else
			check_error(clSetKernelArg(lstm_forget_gate_gradient_kernel, 1, sizeof(cl_mem), &l->cell_state[t]), "lstm fgate grad kernel arg 1");
		check_error(clSetKernelArg(lstm_forget_gate_gradient_kernel, 2, sizeof(cl_mem), &l->forget_gate_output[t]), "lstm fgate grad kernel arg 2");
		check_error(clSetKernelArg(lstm_forget_gate_gradient_kernel, 3, sizeof(cl_mem), &l->forget_gate_gradient[t]), "lstm fgate grad kernel arg 3");
		check_error(clSetKernelArg(lstm_forget_gate_gradient_kernel, 4, sizeof(Nonlinearity), &gate_fn), "lstm fgate grad kernel arg 4");
		check_error(clSetKernelArg(lstm_forget_gate_gradient_kernel, 5, sizeof(int), &use_past_outputs), "lstm fgate grad kernel arg 5");
		check_error(clEnqueueNDRangeKernel(get_opencl_queue0(), lstm_forget_gate_gradient_kernel, 1, NULL, &l->size, NULL, 0, NULL, NULL), "couldn't use fgate grad kernel");

		/* calculate output gate gradient */
		check_error(clSetKernelArg(lstm_output_gate_gradient_kernel, 0, sizeof(cl_mem), &grad[t]), "lstm ogate grad kernel arg 0");
		check_error(clSetKernelArg(lstm_output_gate_gradient_kernel, 1, sizeof(cl_mem), &l->cell_state[t]), "lstm ogate grad kernel arg 0");
		check_error(clSetKernelArg(lstm_output_gate_gradient_kernel, 2, sizeof(cl_mem), &l->output_gate_output[t]), "lstm ogate grad kernel arg 0");
		if(use_future_grads)
			check_error(clSetKernelArg(lstm_output_gate_gradient_kernel, 3, sizeof(cl_mem), &l->input_gradient[t+1]), "lstm ogate grad kernel arg 0");
		else
			check_error(clSetKernelArg(lstm_output_gate_gradient_kernel, 3, sizeof(cl_mem), &l->input_gradient[t]), "lstm ogate grad kernel arg 0");
		check_error(clSetKernelArg(lstm_output_gate_gradient_kernel, 4, sizeof(cl_mem), &l->output_gate_gradient[t]), "lstm ogate grad kernel arg 0");
		check_error(clSetKernelArg(lstm_output_gate_gradient_kernel, 5, sizeof(Nonlinearity), &gate_fn), "lstm ogate grad kernel arg 0");
		check_error(clSetKernelArg(lstm_output_gate_gradient_kernel, 6, sizeof(int), &recurrent_offset), "lstm ogate grad kernel arg 0");
		check_error(clSetKernelArg(lstm_output_gate_gradient_kernel, 7, sizeof(int), &use_future_grads), "lstm ogate grad kernel arg 0");
		check_error(clEnqueueNDRangeKernel(get_opencl_queue0(), lstm_output_gate_gradient_kernel, 1, NULL, &l->size, NULL, 0, NULL, NULL), "couldn't use ogate grad kernel");

		/* calculate gradient of layer inputs (including recurrent inputs */
		check_error(clSetKernelArg(lstm_input_gradient_kernel, 0, sizeof(cl_mem), &l->input_nonl_gradient[t]), "lstm input grad arg 0");
		check_error(clSetKernelArg(lstm_input_gradient_kernel, 1, sizeof(cl_mem), &l->input_gate_gradient[t]), "lstm input grad arg 1");
		check_error(clSetKernelArg(lstm_input_gradient_kernel, 2, sizeof(cl_mem), &l->forget_gate_gradient[t]), "lstm input grad arg 2");
		check_error(clSetKernelArg(lstm_input_gradient_kernel, 3, sizeof(cl_mem), &l->output_gate_gradient[t]), "lstm input grad arg 3");
		check_error(clSetKernelArg(lstm_input_gradient_kernel, 4, sizeof(cl_mem), &params), "lstm input grad arg 4");
		check_error(clSetKernelArg(lstm_input_gradient_kernel, 5, sizeof(cl_mem), &l->input_gradient[t]), "lstm input grad arg 5");
		check_error(clSetKernelArg(lstm_input_gradient_kernel, 6, sizeof(int), &l->size), "lstm input grad arg 6");
		check_error(clSetKernelArg(lstm_input_gradient_kernel, 7, sizeof(int), &l->input_dimension), "lstm input grad arg 7");
		check_error(clSetKernelArg(lstm_input_gradient_kernel, 8, sizeof(int), &l->param_offset), "lstm input grad arg 8");
		check_error(clSetKernelArg(lstm_input_gradient_kernel, 9, sizeof(int), &params_per_cell), "lstm input grad arg 9");
		check_error(clEnqueueNDRangeKernel(get_opencl_queue0(), lstm_input_gradient_kernel, 1, NULL, &l->input_dimension, NULL, 0, NULL, NULL), "couldn't use input gradient kernel");

		/* calculate gradient of layer parameters */
		check_error(clSetKernelArg(lstm_parameter_gradient_kernel, 0, sizeof(cl_mem), &l->input_nonl_gradient[t]), "lstm input grad arg 0");
		check_error(clSetKernelArg(lstm_parameter_gradient_kernel, 1, sizeof(cl_mem), &l->input_gate_gradient[t]), "lstm input grad arg 1");
		check_error(clSetKernelArg(lstm_parameter_gradient_kernel, 2, sizeof(cl_mem), &l->forget_gate_gradient[t]), "lstm input grad arg 2");
		check_error(clSetKernelArg(lstm_parameter_gradient_kernel, 3, sizeof(cl_mem), &l->output_gate_gradient[t]), "lstm input grad arg 3");
		if(use_future_grads){
			check_error(clSetKernelArg(lstm_parameter_gradient_kernel, 4, sizeof(cl_mem), &l->input_nonl_gradient[t+1]), "lstm input grad arg 4");
			check_error(clSetKernelArg(lstm_parameter_gradient_kernel, 5, sizeof(cl_mem), &l->input_gate_gradient[t+1]), "lstm input grad arg 5");
			check_error(clSetKernelArg(lstm_parameter_gradient_kernel, 6, sizeof(cl_mem), &l->forget_gate_gradient[t+1]), "lstm input grad arg 6");
			check_error(clSetKernelArg(lstm_parameter_gradient_kernel, 7, sizeof(cl_mem), &l->output_gate_gradient[t+1]), "lstm input grad arg 7");
		}else{
			check_error(clSetKernelArg(lstm_parameter_gradient_kernel, 4, sizeof(cl_mem), &l->input_nonl_gradient[t]), "lstm input grad arg 4");
			check_error(clSetKernelArg(lstm_parameter_gradient_kernel, 5, sizeof(cl_mem), &l->input_gate_gradient[t]), "lstm input grad arg 5");
			check_error(clSetKernelArg(lstm_parameter_gradient_kernel, 6, sizeof(cl_mem), &l->forget_gate_gradient[t]), "lstm input grad arg 6");
			check_error(clSetKernelArg(lstm_parameter_gradient_kernel, 7, sizeof(cl_mem), &l->output_gate_gradient[t]), "lstm input grad arg 7");
		}
		check_error(clSetKernelArg(lstm_parameter_gradient_kernel, 8,  sizeof(cl_mem), &param_grad), "lstm input grad arg 8");
		check_error(clSetKernelArg(lstm_parameter_gradient_kernel, 9,  sizeof(cl_mem), &l->input[t]), "lstm input grad arg 9");
		check_error(clSetKernelArg(lstm_parameter_gradient_kernel, 10, sizeof(cl_mem), &l->output[t]), "lstm input grad arg 10");
		check_error(clSetKernelArg(lstm_parameter_gradient_kernel, 11, sizeof(int), &use_future_grads), "lstm input grad arg 11");
		check_error(clSetKernelArg(lstm_parameter_gradient_kernel, 12, sizeof(int), &l->size), "lstm input grad arg 12");
		check_error(clSetKernelArg(lstm_parameter_gradient_kernel, 13, sizeof(int), &l->input_dimension), "lstm input grad arg 13");
		check_error(clSetKernelArg(lstm_parameter_gradient_kernel, 14, sizeof(int), &l->param_offset), "lstm input grad arg 14");
		check_error(clSetKernelArg(lstm_parameter_gradient_kernel, 15, sizeof(int), &params_per_cell), "lstm input grad arg 15");
		check_error(clEnqueueNDRangeKernel(get_opencl_queue0(), lstm_parameter_gradient_kernel, 1, NULL, &l->size, NULL, 0, NULL, NULL), "couldn't use parameter gradient kernel");
	}
}

static void gpu_lstm_backward(LSTM *n){
	if(n->t >= n->seq_len){
		cl_mem *grads = n->network_gradient;
		for(int i = n->depth-1; i >= 0; i--){
			LSTM_layer *l = &n->layers[i];
			gpu_lstm_layer_backward(l, grads, n->params, n->param_grad, n->t-1);
			grads = n->layers[i].input_gradient;
		}
		/*
		for(int i = n->depth-1; i >= 0; i--){
			gpu_zero_2d_arr(n->layers[i].input_gradient, MAX_UNROLL_LENGTH, n->layers[i].input_dimension);
			gpu_zero_2d_arr(n->layers[i].output, MAX_UNROLL_LENGTH, n->layers[i].size);
		}
		*/
		check_error(clFinish(get_opencl_queue0()), "waiting for kernels to finish executing (backward pass)");
		if(!n->stateful)
			gpu_wipe(n);
		n->t = 0;
	}
}

#endif

LSTM lstm_from_arr(size_t *arr, size_t len){
#ifdef GPU
	return gpu_lstm_from_arr(arr, len);
#else
	return cpu_lstm_from_arr(arr, len);
#endif
}

void lstm_forward(LSTM *n, float *x){
#ifdef GPU
	gpu_lstm_forward(n, x);
#else
	cpu_lstm_forward(n, x);
#endif
}

void lstm_backward(LSTM *n){
#ifdef GPU
	gpu_lstm_backward(n);
#else
	cpu_lstm_backward(n);
#endif
}

void wipe(LSTM *n){
#ifdef GPU
	gpu_wipe(n);
#else
	cpu_wipe(n);
#endif
}

void dealloc_lstm(LSTM *n){
	/*
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
			free(c->gradient);
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
	*/
}
 /*
  * IO FUNCTIONS FOR READING AND WRITING TO A FILE
  */

static int getWord(FILE *fp, char* dest){
  memset(dest, '\0', strlen(dest));
  return fscanf(fp, " %1023s", dest);
}

void save_lstm(LSTM *n, const char *filename){
	FILE *fp = fopen(filename, "w");
	if(!fp){
		printf("ERROR: save_lstm(): could not open file '%s' (correct filepath? does dir exist?)", filename);
		exit(1);
	}
	fprintf(fp, "LSTM %lu %lu ", n->depth, n->input_dimension);
	for(int i = 0; i < n->depth; i++){
		fprintf(fp, "%lu", n->layers[i].size);
		fprintf(fp, " ");
	}
	fprintf(fp, "%lu\n", n->output_layer.size);

#ifdef GPU
	float *tmp = (float*)malloc(sizeof(float)*n->num_params);
	check_error(clEnqueueReadBuffer(get_opencl_queue0(), n->params, 1, 0, sizeof(float) * n->num_params, tmp, 0, NULL, NULL), "error reading params from gpu");
	for(int i = 0; i < n->num_params; i++){
		fprintf(fp, "%f", tmp[i]);
		if(i < n->num_params-1) fprintf(fp, " ");
		else fprintf(fp, "\n");
	}
	free(tmp);
#else
	for(int i = 0; i < n->num_params; i++){
		fprintf(fp, "%f", n->params[i]);
		if(i < n->num_params-1) fprintf(fp, " ");
		else fprintf(fp, "\n");
	}
#endif
	fclose(fp);
}

LSTM load_lstm(const char *filename){
  FILE *fp = fopen(filename, "rb");
  char buff[1024];
  memset(buff, '\0', 1024);

  getWord(fp, buff); //Get first word to check if MLP file

  if(strcmp(buff, "LSTM") != 0){
    printf("ERROR: [%s] is not an LSTM.\n", buff);
    exit(1);
  }
	size_t num_layers, input_dim;
	if(fscanf(fp, "%lu %lu", &num_layers, &input_dim) == EOF){
		printf("ERROR: '%s' potentially corrupted.\n", filename);
		exit(1);
	}

	size_t arr[num_layers+2];
	arr[0] = input_dim;
	for(int i = 1; i <= num_layers; i++){
		if(fscanf(fp, " %lu", &arr[i]) == EOF){
			printf("ERROR: '%s' potentially corrupted.\n", filename);
			exit(1);
		}
	}
	if(fscanf(fp, " %lu", &arr[num_layers+1]) == EOF){
		printf("ERROR: '%s' potentially corrupted.\n", filename);
		exit(1);
	}
	LSTM n = lstm_from_arr(arr, num_layers+2);
#ifndef GPU
	for(int i = 0; i < n.num_params; i++){
		if(fscanf(fp, "%f", &n.params[i]) == EOF){
			printf("ERROR: '%s' potentially corrupted.\n", filename);
			exit(1);
		}
	}
#else
	float *tmp = (float*)malloc(sizeof(float)*n.num_params);
	for(int i = 0; i < n.num_params; i++){
		if(fscanf(fp, "%f", &tmp[i]) == EOF){
			printf("ERROR: '%s' potentially corrupted.\n", filename);
			exit(1);
  	}
	}
	check_error(clEnqueueWriteBuffer(get_opencl_queue0(), n.params, 1, 0, sizeof(float) * n.num_params, tmp, 0, NULL, NULL), "copying input");
	free(tmp);
#endif
	return n;
}
