/* Author: Jonah Siekmann
 * 10/3/2018
 * This is a simple implementation of a Long Short-Term Memory (LSTM) network, as described by Sepp Hochreiter and Jurgen Schmidhuber in their 1993 paper, with the addition of the forget gate described by Felix Gers.
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

#define PRINTLIST(name, len) printf("printing %s: [", #name); for(int xyz = 0; xyz < len; xyz++){printf("%5.4f", name[xyz]); if(xyz < len-1) printf(", "); else printf("]\n");}
#define ALLOCATE(TYPE, NUM) (TYPE*)malloc((NUM) * (sizeof(TYPE)));
#define DEBUG 1
#define MAX_GRAD 1
#define MAX_STATE 35

/*
 * Used to sample from softmax distribution
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

int argmax(float *args, size_t len){
	int argm = 0;
	for(int i = 0; i < len; i++){
		if(args[argm] < args[i])
			argm = i;
	}
	return argm;
}

float lstm_cost(LSTM *n, float *y){
	MLP_layer *mlp = &n->output_layer;
	float tmp[n->output_dimension];
	float c = n->cost_fn(n->output, y, tmp, n->output_dimension);
#ifndef GPU
	cpu_mlp_layer_backward(mlp, tmp);
	float *grads = mlp->gradient;
#else
	//check_error(clEnqueueWriteBuffer(get_opencl_queue(), n->network_input, 0, 0, sizeof(float) * n->input_dimension, x, 0, NULL, NULL), "enqueuing network input");
	printf("need to implement gpu cost\n");
	//cl_mem grads = mlp->layers[0].gradient;
#endif
	
	size_t t = n->t;
	if(mlp->input_dimension != n->layers[n->depth-1].size){
		printf("ERROR: cost_wrapper(): Size mismatch between mlp output layer and final lstm layer (%lu vs %lu)\n", mlp->input_dimension, n->layers[n->depth-1].size);
		exit(1);
	}
#ifndef GPU
	for(int i = 0; i < mlp->input_dimension; i++)
		n->network_gradient[t][i] = grads[i];
#endif
	

	n->t++;
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
	g.dOutput = ALLOCATE(float, sequence_length);
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
		*forget_gate_bias = 1.0;
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

		a->output[t] = HYPERTAN(inner_product(a->weights, tmp, l->input_dimension) + *a->bias); //input nonlinearity uses hypertangent
		i->output[t] = SIGMOID(inner_product(i->weights, tmp, l->input_dimension) + *i->bias); //all of the gates use sigmoid
		f->output[t] = SIGMOID(inner_product(f->weights, tmp, l->input_dimension) + *f->bias);
		o->output[t] = SIGMOID(inner_product(o->weights, tmp, l->input_dimension) + *o->bias);

		a->dOutput[t] = D_HYPERTAN(a->output[t]);
		i->dOutput[t] = D_SIGMOID(i->output[t]);
		f->dOutput[t] = D_SIGMOID(f->output[t]);
		o->dOutput[t] = D_SIGMOID(o->output[t]);

		c->state[t] = a->output[t] * i->output[t] + f->output[t] * c->lstate; //Calculate the internal cell state
		l->output[t][j] = HYPERTAN(c->state[t]) * o->output[t]; //Calculate the output of the cell

#if DEBUG
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

		if(c->lstate > MAX_STATE) c->lstate = MAX_STATE;
		if(c->lstate < -MAX_STATE) c->lstate = -MAX_STATE;
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
			float grad = grads[t][j]; 
			c->dOutput[t] = grad + delta_out;
			c->dstate[t] = c->dOutput[t] * o->output[t] * D_HYPERTAN(HYPERTAN(c->state[t])) + next_dstate * next_forget;

			a->gradient[t] = c->dstate[t] * i->output[t] * a->dOutput[t];
			i->gradient[t] = c->dstate[t] * a->output[t] * i->dOutput[t];
			if(t) f->gradient[t] = c->dstate[t] * c->state[t-1] * f->dOutput[t];
			else  f->gradient[t] = 0;
			o->gradient[t] = c->dOutput[t] * HYPERTAN(c->state[t]) * o->output[t] * (1 - o->output[t]);

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
					if(isnan(a->weights[k])){
						printf("ERROR: layer_backward(): nan'ed a weight grad, from %6.5f * %6.5f\n", a->gradient[t+1], l->output[t][j]);
						exit(1);
					}
				}
			}
			*a->bias_grad += a->gradient[t];
			*i->bias_grad += i->gradient[t];
			*f->bias_grad += f->gradient[t];
			*o->bias_grad += o->gradient[t];
		}
		//clip gradients that are too large (to put a band-aid on exploding gradients)
		for(long j = 0; j < l->input_dimension; j++){
			if(l->input_gradient[t][j] > MAX_GRAD) l->input_gradient[t][j] = MAX_GRAD;
			if(l->input_gradient[t][j] < -MAX_GRAD) l->input_gradient[t][j] = -MAX_GRAD;
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

cl_kernel mlp_forward_kernel, mlp_backward_kernel;
cl_kernel lstm_forward_kernel, lstm_backward_kernel;
cl_kernel logistic_kernel;

static int ARE_KERNELS_INITIALIZED = 0;
void lstm_kernel_setup(){
	mlp_kernel_setup();

	char *kernels[] = {"include/nonlinear.h", "src/lstm.cl", "src/mlp.cl", "src/logistic.cl"};

	int err = 0;

	char *src = get_kernel_source(kernels, 4);
	cl_program prog = build_program(src);
	free(src);

	mlp_forward_kernel = clCreateKernel(prog, "mlp_forward_kernel", &err);
	check_error(err, "couldn't make linear forward kernel");

	logistic_kernel = clCreateKernel(prog, "logistic_kernel", &err);
	check_error(err, "couldn't make logistic kernel");

	//lstm_forward_kernel = clCreateKernel(prog, "lstm_forward_kernel", &err);
	//check_error(&err, "couldn't make linear forward kernel");
	
}

void gpu_zero_2d_arr(cl_mem *arr, size_t arrs){

}

void gpu_wipe(LSTM *n){

}

LSTM_layer gpu_create_LSTM_layer(size_t input_dim, size_t size, int param_offset){
	LSTM_layer l;
	l.size = size;
	l.param_offset = param_offset;
	l.input_dimension = input_dim;

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

	l.cell_state = ALLOCATE(cl_mem, MAX_UNROLL_LENGTH);

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
	}
	l.cell_state = ALLOCATE(cl_mem, MAX_UNROLL_LENGTH);

	l.input  = ALLOCATE(cl_mem, MAX_UNROLL_LENGTH);
	l.output = ALLOCATE(cl_mem, MAX_UNROLL_LENGTH);
	return l;
}

LSTM gpu_lstm_from_arr(size_t *arr, size_t len){
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
	n.params = ALLOCATE(float, num_params);
	
	int err;
	n.gpu_params = clCreateBuffer(get_opencl_context(), CL_MEM_READ_WRITE, sizeof(float) * n.num_params, NULL, &err);
	check_error(err, "creating gpu params");
	n.param_grad = clCreateBuffer(get_opencl_context(), CL_MEM_READ_WRITE, sizeof(float) * n.num_params, NULL, &err);
	check_error(err, "creating param grad");

	n.network_gradient = ALLOCATE(cl_mem, MAX_UNROLL_LENGTH);
	n.network_input    = ALLOCATE(cl_mem, MAX_UNROLL_LENGTH);

	for(int i = 0; i < MAX_UNROLL_LENGTH; i++){
		n.network_gradient[i] = clCreateBuffer(get_opencl_context(), CL_MEM_READ_WRITE, sizeof(float) * arr[len-2], NULL, &err);
		n.network_input[i] = clCreateBuffer(get_opencl_context(), CL_MEM_READ_WRITE, sizeof(float) * arr[0], NULL, &err);
	}

	gpu_zero_2d_arr(n.network_gradient, MAX_UNROLL_LENGTH);
	gpu_zero_2d_arr(n.network_input, MAX_UNROLL_LENGTH);
	
	int param_idx = 0;
	for(int i = 1; i < len-1; i++){
		LSTM_layer l = gpu_create_LSTM_layer(arr[i-1], arr[i], param_idx);
		param_idx += (4*(arr[i-1]+arr[i]+1))*arr[i];
		n.layers[i-1] = l;
	}

	return n;
}

/* oh boy here we go */
static void gpu_lstm_layer_forward(LSTM_layer *l, cl_mem x, cl_mem params, size_t t){
	l->input[t] = x;

	Nonlinearity gate_fn = sigmoid;
	Nonlinearity nonl_fn = hypertan;

	int params_per_gate  = (l->input_dimension+l->size+1);
	int input_nonl_base  = l->param_offset + 0 * params_per_gate;
	int input_gate_base  = l->param_offset + 1 * params_per_gate;
	int forget_gate_base = l->param_offset + 2 * params_per_gate;
	int output_gate_base = l->param_offset + 3 * params_per_gate;
	int skipdist         =                   4 * params_per_gate;

	check_error(clSetKernelArg(mlp_forward_kernel, 0, sizeof(cl_mem), &x), "setting forward kernel arg0");
	check_error(clSetKernelArg(mlp_forward_kernel, 1, sizeof(cl_mem), &l->input_nonl_z), "setting forward kernel arg1");
	check_error(clSetKernelArg(mlp_forward_kernel, 2, sizeof(cl_mem), &params), "setting forward kernel arg2");
	check_error(clSetKernelArg(mlp_forward_kernel, 3, sizeof(int), &l->input_dimension), "setting forward kernel arg3");
	check_error(clSetKernelArg(mlp_forward_kernel, 4, sizeof(int), &input_nonl_base), "setting forward kernel arg4");
	check_error(clSetKernelArg(mlp_forward_kernel, 5, sizeof(int), &skipdist), "setting forward kernel arg5");
	check_error(clEnqueueNDRangeKernel(get_opencl_queue(), mlp_forward_kernel, 1, NULL, &l->size, NULL, 0, NULL, NULL), "couldn't enqueue linear kernel");
	
	check_error(clSetKernelArg(mlp_forward_kernel, 0, sizeof(cl_mem), &x), "setting forward kernel arg0");
	check_error(clSetKernelArg(mlp_forward_kernel, 1, sizeof(cl_mem), &l->input_gate_z), "setting forward kernel arg1");
	check_error(clSetKernelArg(mlp_forward_kernel, 2, sizeof(cl_mem), &params), "setting forward kernel arg2");
	check_error(clSetKernelArg(mlp_forward_kernel, 3, sizeof(int), &l->input_dimension), "setting forward kernel arg3");
	check_error(clSetKernelArg(mlp_forward_kernel, 4, sizeof(int), &input_gate_base), "setting forward kernel arg4");
	check_error(clSetKernelArg(mlp_forward_kernel, 5, sizeof(int), &skipdist), "setting forward kernel arg5");
	check_error(clEnqueueNDRangeKernel(get_opencl_queue(), mlp_forward_kernel, 1, NULL, &l->size, NULL, 0, NULL, NULL), "couldn't enqueue linear kernel");

	check_error(clSetKernelArg(mlp_forward_kernel, 0, sizeof(cl_mem), &x), "setting forward kernel arg0");
	check_error(clSetKernelArg(mlp_forward_kernel, 1, sizeof(cl_mem), &l->forget_gate_z), "setting forward kernel arg1");
	check_error(clSetKernelArg(mlp_forward_kernel, 2, sizeof(cl_mem), &params), "setting forward kernel arg2");
	check_error(clSetKernelArg(mlp_forward_kernel, 3, sizeof(int), &l->input_dimension), "setting forward kernel arg3");
	check_error(clSetKernelArg(mlp_forward_kernel, 4, sizeof(int), &forget_gate_base), "setting forward kernel arg4");
	check_error(clSetKernelArg(mlp_forward_kernel, 5, sizeof(int), &skipdist), "setting forward kernel arg5");
	check_error(clEnqueueNDRangeKernel(get_opencl_queue(), mlp_forward_kernel, 1, NULL, &l->size, NULL, 0, NULL, NULL), "couldn't enqueue linear kernel");

	check_error(clSetKernelArg(mlp_forward_kernel, 0, sizeof(cl_mem), &x), "setting forward kernel arg0");
	check_error(clSetKernelArg(mlp_forward_kernel, 1, sizeof(cl_mem), &l->output_gate_z), "setting forward kernel arg1");
	check_error(clSetKernelArg(mlp_forward_kernel, 2, sizeof(cl_mem), &params), "setting forward kernel arg2");
	check_error(clSetKernelArg(mlp_forward_kernel, 3, sizeof(int), &l->input_dimension), "setting forward kernel arg3");
	check_error(clSetKernelArg(mlp_forward_kernel, 4, sizeof(int), &output_gate_base), "setting forward kernel arg4");
	check_error(clSetKernelArg(mlp_forward_kernel, 5, sizeof(int), &skipdist), "setting forward kernel arg5");
	check_error(clEnqueueNDRangeKernel(get_opencl_queue(), mlp_forward_kernel, 1, NULL, &l->size, NULL, 0, NULL, NULL), "couldn't enqueue linear kernel");

	check_error(clSetKernelArg(logistic_kernel, 0, sizeof(cl_mem), &l->input_nonl_z), "setting logistic arg 0");
	check_error(clSetKernelArg(logistic_kernel, 1, sizeof(cl_mem), &l->input_nonl_output), "setting logistic arg 0");
	check_error(clSetKernelArg(logistic_kernel, 2, sizeof(Nonlinearity), &nonl_fn), "setting logistic arg 0");
	check_error(clEnqueueNDRangeKernel(get_opencl_queue(), logistic_kernel, 1, NULL, &l->size, NULL, 0, NULL, NULL), "couldn't enqueue logistic kernel");

	check_error(clSetKernelArg(logistic_kernel, 0, sizeof(cl_mem), &l->input_gate_z), "setting logistic arg 0");
	check_error(clSetKernelArg(logistic_kernel, 1, sizeof(cl_mem), &l->input_gate_output), "setting logistic arg 0");
	check_error(clSetKernelArg(logistic_kernel, 2, sizeof(Nonlinearity), &gate_fn), "setting logistic arg 0");
	check_error(clEnqueueNDRangeKernel(get_opencl_queue(), logistic_kernel, 1, NULL, &l->size, NULL, 0, NULL, NULL), "couldn't enqueue logistic kernel");

	check_error(clSetKernelArg(logistic_kernel, 0, sizeof(cl_mem), &l->forget_gate_z), "setting logistic arg 0");
	check_error(clSetKernelArg(logistic_kernel, 1, sizeof(cl_mem), &l->forget_gate_output), "setting logistic arg 0");
	check_error(clSetKernelArg(logistic_kernel, 2, sizeof(Nonlinearity), &gate_fn), "setting logistic arg 0");
	check_error(clEnqueueNDRangeKernel(get_opencl_queue(), logistic_kernel, 1, NULL, &l->size, NULL, 0, NULL, NULL), "couldn't enqueue logistic kernel");

	check_error(clSetKernelArg(logistic_kernel, 0, sizeof(cl_mem), &l->output_gate_z), "setting logistic arg 0");
	check_error(clSetKernelArg(logistic_kernel, 1, sizeof(cl_mem), &l->output_gate_output), "setting logistic arg 0");
	check_error(clSetKernelArg(logistic_kernel, 2, sizeof(Nonlinearity), &gate_fn), "setting logistic arg 0");
	check_error(clEnqueueNDRangeKernel(get_opencl_queue(), logistic_kernel, 1, NULL, &l->size, NULL, 0, NULL, NULL), "couldn't enqueue logistic kernel");
}

static void gpu_lstm_forward(LSTM *n, float *x){
	size_t t = n->t;
	check_error(clEnqueueWriteBuffer(get_opencl_queue(), n->network_input[t], 0, 0, sizeof(float) * n->input_dimension, x, 0, NULL, NULL), "copying input");

	//Feedforward through all LSTM layers
	cl_mem input = n->network_input[t];
	for(int i = 0; i < n->depth; i++){
		LSTM_layer *l = &n->layers[i];
		gpu_lstm_layer_forward(l, input, n->gpu_params, t);
		input = l->output[t];
	}

	//Feedforward through final MLP layer
	gpu_mlp_layer_forward(&n->output_layer, input, n->gpu_params);

	check_error(clEnqueueReadBuffer(get_opencl_queue(), n->output_layer.output, 1, 0, sizeof(float) * n->output_dimension, n->output, 0, NULL, NULL), "error reading from gpu");

  if(!(rand()%3))
		n->guess = sample_softmax(n->output, n->output_dimension);
	else
		n->guess = argmax(n->output, n->output_dimension);

}

static void gpu_lstm_backward(LSTM *n){

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
	*/
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
	fprintf(fp, "%lu\n", n->output_layer.size);

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
