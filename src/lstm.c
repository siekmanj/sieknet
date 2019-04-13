/* Author: Jonah Siekmann
 * 10/3/2018
 * This is a simple implementation of a Long Short-Term Memory (LSTM) network, as described by Sepp Hochreiter and Jurgen Schmidhuber in their 1997 paper, with the addition of the forget gate described by Felix Gers.
 * Big thanks to Aidan Gomez for his wonderful numerical example of the backpropagation algorithm for an LSTM cell:
 * https://blog.aidangomez.ca/2016/04/17/Backpropogating-an-LSTM-A-Numerical-Example/
 */

#include <lstm.h>
#include <rnn.h>
#include <logistic.h>
#include <math.h>
#include <string.h>

#ifdef SIEKNET_USE_GPU
#include "opencl_utils.h"
static cl_kernel make_onehot_kernel;
static cl_kernel rnn_forward_kernel;
static cl_kernel lstm_forward_kernel;
static cl_kernel lstm_input_gradient_kernel, lstm_parameter_gradient_kernel;
static cl_kernel lstm_input_nonl_gradient_kernel, lstm_forget_gate_gradient_kernel, lstm_output_gate_gradient_kernel, lstm_dstate_kernel;
static cl_kernel logistic_kernel, zero_init_kernel;

static int ARE_KERNELS_INITIALIZED = 0;
void lstm_kernel_setup(){
  mlp_kernel_setup();

  char *kernels[] = {"include/conf.h", "include/logistic.h", "include/rnn.h", "include/lstm.h", "src/logistic.cl", "src/rnn.cl", "src/lstm.cl"};

  int err = 0;

  char *src = get_kernel_source(kernels, sizeof(kernels)/sizeof(kernels[0]));
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
#endif

#define ARR_FROM_GPU(name, gpumem, size) float name[size]; memset(name, '\0', size*sizeof(float)); check_error(clEnqueueReadBuffer(get_opencl_queue0(), gpumem, 1, 0, sizeof(float) * size, name, 0, NULL, NULL), "error reading from gpu (ARR_FROM_SIEKNET_USE_GPU)");


/*
 * Used to reset the hidden state of the lstm
 */ 
#ifndef SIEKNET_USE_GPU
static void cpu_lstm_wipe(LSTM *n){
  for(int i = 0; i < n->depth; i++){
    LSTM_layer *l = &n->layers[i];
    cpu_zero_2d_arr(&l->loutput, 1, l->size);
    cpu_zero_2d_arr(&l->lstate, 1, l->size);
  }
  cpu_zero_2d_arr(n->recurrent_gradient, SIEKNET_MAX_UNROLL_LENGTH, n->layers[n->depth-1].size);
  n->t = 0;
}	
#else
void gpu_lstm_wipe(LSTM *n){
  gpu_zero_2d_arr(n->recurrent_gradient, SIEKNET_MAX_UNROLL_LENGTH, n->input_dimension);
  //gpu_zero_2d_arr(n->network_input,    SIEKNET_MAX_UNROLL_LENGTH, n->input_dimension);

  for(int i = 0; i < n->depth; i++){
    LSTM_layer *l = &n->layers[i];
    check_error(clSetKernelArg(zero_init_kernel, 0, sizeof(cl_mem), &l->loutput), "couldn't set zero kernel arg 0");
    check_error(clEnqueueNDRangeKernel(get_opencl_queue0(), zero_init_kernel, 1, NULL, &l->size, NULL, 0, NULL, NULL), "couldn't use zero kernel");
    check_error(clSetKernelArg(zero_init_kernel, 0, sizeof(cl_mem), &l->lstate), "couldn't set zero kernel arg 0");
    check_error(clEnqueueNDRangeKernel(get_opencl_queue0(), zero_init_kernel, 1, NULL, &l->size, NULL, 0, NULL, NULL), "couldn't use zero kernel");
  }
}
#endif




/*
 * Used to initialize & allocate memory for a layer of LSTM cells
 */
#ifndef SIEKNET_USE_GPU
LSTM_layer cpu_create_LSTM_layer(size_t input_dim, size_t size, float *params, int param_offset){
  LSTM_layer l;
  l.size = size;
  l.param_offset = param_offset;
  l.input_dimension = input_dim + size;

  size_t neuron_offset = param_offset;
  for(int i = 0; i < size*4; i++){
    xavier_init(&params[neuron_offset], (l.input_dimension+1), l.size);
    neuron_offset += l.input_dimension+1;
  }

  l.input_nonl_z  = alloc_2d_array(SIEKNET_MAX_UNROLL_LENGTH, l.size);
  l.input_gate_z  = alloc_2d_array(SIEKNET_MAX_UNROLL_LENGTH, l.size);
  l.forget_gate_z = alloc_2d_array(SIEKNET_MAX_UNROLL_LENGTH, l.size);
  l.output_gate_z = alloc_2d_array(SIEKNET_MAX_UNROLL_LENGTH, l.size);

  l.input_nonl_output  = alloc_2d_array(SIEKNET_MAX_UNROLL_LENGTH, l.size);
  l.input_gate_output  = alloc_2d_array(SIEKNET_MAX_UNROLL_LENGTH, l.size);
  l.forget_gate_output = alloc_2d_array(SIEKNET_MAX_UNROLL_LENGTH, l.size);
  l.output_gate_output = alloc_2d_array(SIEKNET_MAX_UNROLL_LENGTH, l.size);

  l.input_nonl_gradient  = alloc_2d_array(SIEKNET_MAX_UNROLL_LENGTH, l.size);
  l.input_gate_gradient  = alloc_2d_array(SIEKNET_MAX_UNROLL_LENGTH, l.size);
  l.forget_gate_gradient = alloc_2d_array(SIEKNET_MAX_UNROLL_LENGTH, l.size);
  l.output_gate_gradient = alloc_2d_array(SIEKNET_MAX_UNROLL_LENGTH, l.size);

  l.input_gradient = alloc_2d_array(SIEKNET_MAX_UNROLL_LENGTH, l.input_dimension);
  l.cell_gradient = alloc_2d_array(SIEKNET_MAX_UNROLL_LENGTH, l.size);

  l.cell_state = alloc_2d_array(SIEKNET_MAX_UNROLL_LENGTH, l.size);
  l.cell_dstate = alloc_2d_array(SIEKNET_MAX_UNROLL_LENGTH, l.size);

  l.input = ALLOC(float*, SIEKNET_MAX_UNROLL_LENGTH);
  l.output = alloc_2d_array(SIEKNET_MAX_UNROLL_LENGTH, l.size);

  l.loutput = ALLOC(float, l.size);
  l.lstate  = ALLOC(float, l.size);

  cpu_zero_2d_arr(l.input_gradient, SIEKNET_MAX_UNROLL_LENGTH, l.input_dimension);

  return l;
}
#else
LSTM_layer gpu_create_LSTM_layer(size_t input_dim, size_t size, cl_mem params, int param_offset){
  LSTM_layer l;
  l.size = size;
  l.param_offset = param_offset;
  l.input_dimension = input_dim + size;

  size_t neuron_offset = 0;
  float *tmp = (float*)malloc(sizeof(float)*l.size*(l.input_dimension+1)*4);
  for(int i = 0; i < size*4; i++){
    xavier_init(&tmp[neuron_offset], (l.input_dimension+1), l.size);
    neuron_offset += l.input_dimension+1;
  }
  check_error(clEnqueueWriteBuffer(get_opencl_queue0(), params, 1, sizeof(float) * param_offset, sizeof(float)*l.size*(l.input_dimension+1)*4, tmp, 0, NULL, NULL), "could not enqueue layer params");
  free(tmp);

  l.input_nonl_z  = ALLOC(cl_mem, SIEKNET_MAX_UNROLL_LENGTH);
  l.input_gate_z  = ALLOC(cl_mem, SIEKNET_MAX_UNROLL_LENGTH);
  l.forget_gate_z = ALLOC(cl_mem, SIEKNET_MAX_UNROLL_LENGTH);
  l.output_gate_z = ALLOC(cl_mem, SIEKNET_MAX_UNROLL_LENGTH);

  l.input_nonl_output  = ALLOC(cl_mem, SIEKNET_MAX_UNROLL_LENGTH);
  l.input_gate_output  = ALLOC(cl_mem, SIEKNET_MAX_UNROLL_LENGTH);
  l.forget_gate_output = ALLOC(cl_mem, SIEKNET_MAX_UNROLL_LENGTH);
  l.output_gate_output = ALLOC(cl_mem, SIEKNET_MAX_UNROLL_LENGTH);

  l.input_nonl_gradient  = ALLOC(cl_mem, SIEKNET_MAX_UNROLL_LENGTH);
  l.input_gate_gradient  = ALLOC(cl_mem, SIEKNET_MAX_UNROLL_LENGTH);
  l.forget_gate_gradient = ALLOC(cl_mem, SIEKNET_MAX_UNROLL_LENGTH);
  l.output_gate_gradient = ALLOC(cl_mem, SIEKNET_MAX_UNROLL_LENGTH);

  l.cell_state     = ALLOC(cl_mem, SIEKNET_MAX_UNROLL_LENGTH);
  l.cell_dstate    = ALLOC(cl_mem, SIEKNET_MAX_UNROLL_LENGTH);
  l.cell_gradient  = ALLOC(cl_mem, SIEKNET_MAX_UNROLL_LENGTH);
  l.input_gradient = ALLOC(cl_mem, SIEKNET_MAX_UNROLL_LENGTH);

  l.input  = ALLOC(cl_mem, SIEKNET_MAX_UNROLL_LENGTH);
  l.output = ALLOC(cl_mem, SIEKNET_MAX_UNROLL_LENGTH);

  int err;
  for(int t = 0; t < SIEKNET_MAX_UNROLL_LENGTH; t++){
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
  check_error(err, "allocating internal lstm memory");
  l.lstate = clCreateBuffer(get_opencl_context(), CL_MEM_READ_WRITE, sizeof(float) * l.size, NULL, &err); 
  check_error(err, "allocating internal lstm memory");
  gpu_zero_2d_arr(&l.loutput, 1, l.size);
  gpu_zero_2d_arr(&l.lstate, 1, l.size);
  return l;
}
#endif

/*
 * Creates an LSTM for the CPU from an array of layer sizes. Called through a
 * macro create_lstm(...), which allows a variable number of parameters.
 */
#ifndef SIEKNET_USE_GPU
LSTM cpu_lstm_from_arr(size_t *arr, size_t len){
  LSTM n;
  n.t = 0;
  n.stateful = 0;
  n.seq_len = 25;
  n.input_dimension = arr[0];
  n.output_dimension = arr[len-1];
  n.depth = len-2;
  n.cost_fn = cross_entropy;

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

  n.layers = ALLOC(LSTM_layer, len-2);
  n.params = ALLOC(float, num_params);
  n.param_grad = ALLOC(float, num_params);

  memset(n.param_grad, '\0', num_params*sizeof(float));

  int param_idx = 0;
  for(int i = 1; i < len-1; i++){
    LSTM_layer l = cpu_create_LSTM_layer(arr[i-1], arr[i], n.params, param_idx);
    param_idx += (4*(arr[i-1]+arr[i]+1))*arr[i];
    n.layers[i-1] = l;
  }	

  //Allocate the 2d array to store the gradients calculated by the mlp output layer
  n.recurrent_gradient = alloc_2d_array(SIEKNET_MAX_UNROLL_LENGTH, arr[len-2]);

  n.network_input = alloc_2d_array(SIEKNET_MAX_UNROLL_LENGTH, arr[0]);

  n.cost_gradient = ALLOC(float, n.output_dimension);

  cpu_zero_2d_arr(n.recurrent_gradient, SIEKNET_MAX_UNROLL_LENGTH, arr[len-2]);
  cpu_zero_2d_arr(n.network_input, SIEKNET_MAX_UNROLL_LENGTH, arr[0]);

  n.output_layer = cpu_create_MLP_layer(arr[len-2], arr[len-1], n.params, param_idx, softmax);

  n.output = n.output_layer.output;
  return n;
}
#else
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
  n.cost_fn = cross_entropy;

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

  n.layers = ALLOC(LSTM_layer, len-2);

  int err;
  n.params = clCreateBuffer(get_opencl_context(), CL_MEM_READ_WRITE, sizeof(float) * n.num_params, NULL, &err);
  check_error(err, "creating gpu params");
  n.param_grad = clCreateBuffer(get_opencl_context(), CL_MEM_READ_WRITE, sizeof(float) * n.num_params, NULL, &err);
  check_error(err, "creating param grad");

  n.recurrent_gradient = ALLOC(cl_mem, SIEKNET_MAX_UNROLL_LENGTH);
  n.network_input      = ALLOC(cl_mem, SIEKNET_MAX_UNROLL_LENGTH);
  n.cost_gradient      = clCreateBuffer(get_opencl_context(), CL_MEM_READ_WRITE, sizeof(float) * n.output_dimension, NULL, &err);
  check_error(err, "allocating gpu mem for cost grad");
  n.output_label       = clCreateBuffer(get_opencl_context(), CL_MEM_READ_WRITE, sizeof(float) * n.output_dimension, NULL, &err);
  check_error(err, "allocating gpu mem for output label");

  for(int t = 0; t < SIEKNET_MAX_UNROLL_LENGTH; t++){
    n.recurrent_gradient[t] = clCreateBuffer(get_opencl_context(), CL_MEM_READ_WRITE, sizeof(float) * arr[len-2], NULL, &err);
    check_error(err, "couldn't make internal lstm memory (network gradient)");
    n.network_input[t] = clCreateBuffer(get_opencl_context(), CL_MEM_READ_WRITE, sizeof(float) * arr[0], NULL, &err);
    check_error(err, "couldn't make internal lstm memory (network input)");
  }

  gpu_zero_2d_arr(n.recurrent_gradient, SIEKNET_MAX_UNROLL_LENGTH, arr[len-2]);
  gpu_zero_2d_arr(n.network_input, SIEKNET_MAX_UNROLL_LENGTH, arr[0]);
  gpu_zero_2d_arr(&n.param_grad, 1, n.num_params);

  int param_idx = 0;
  for(int i = 1; i < len-1; i++){
    LSTM_layer l = gpu_create_LSTM_layer(arr[i-1], arr[i], n.params, param_idx);
    param_idx += (4*(arr[i-1]+arr[i]+1))*arr[i];
    n.layers[i-1] = l;
  }
  n.output_layer = gpu_create_MLP_layer(arr[len-2], arr[len-1], n.params, param_idx, softmax);
  n.output = ALLOC(float, n.output_dimension);

  gpu_lstm_wipe(&n);
  return n;
}

#endif

/*
 * Computes the forward pass of a single layer
 */
#ifndef SIEKNET_USE_GPU
void cpu_lstm_layer_forward(LSTM_layer *l, float *input, float *params, size_t t){
  l->input[t] = input; /* save pointer to input for backward pass */

  Nonlinearity gate_fn = sigmoid;
  Nonlinearity nonl_fn = hypertan;

  int params_per_gate  = (l->input_dimension+1);
  int input_nonl_base  = l->param_offset + 0 * params_per_gate;
  int input_gate_base  = l->param_offset + 1 * params_per_gate;
  int forget_gate_base = l->param_offset + 2 * params_per_gate;
  int output_gate_base = l->param_offset + 3 * params_per_gate;
  int skipdist         =                   4 * params_per_gate;

  /* do output calculations for every cell in this layer */
  for(int i = 0; i < l->size; i++){

    agnostic_rnn_forward_kernel(input, l->loutput, l->input_nonl_z[t],  params, l->input_dimension, l->size, input_nonl_base, skipdist, i);
    agnostic_rnn_forward_kernel(input, l->loutput, l->input_gate_z[t],  params, l->input_dimension, l->size, input_gate_base, skipdist, i);
    agnostic_rnn_forward_kernel(input, l->loutput, l->forget_gate_z[t], params, l->input_dimension, l->size, forget_gate_base, skipdist, i);
    agnostic_rnn_forward_kernel(input, l->loutput, l->output_gate_z[t], params, l->input_dimension, l->size, output_gate_base, skipdist, i);

    l->input_nonl_output[t][i]  = activate(l->input_nonl_z[t][i], nonl_fn);
    l->input_gate_output[t][i]  = activate(l->input_gate_z[t][i], gate_fn);
    l->forget_gate_output[t][i] = activate(l->forget_gate_z[t][i], gate_fn);
    l->output_gate_output[t][i] = activate(l->output_gate_z[t][i], gate_fn);

    agnostic_lstm_forward_kernel(l->input_nonl_output[t], 
        l->input_gate_output[t], 
        l->forget_gate_output[t], 
        l->output_gate_output[t], 
        l->cell_state[t],
        l->lstate,
        l->output[t],
        i);

  }
  for(int i = 0; i < l->size; i++){
    l->loutput[i] = l->output[t][i];
    l->lstate[i] = l->cell_state[t][i];
  }
}
#else
/* oh boy here we go */
static void gpu_lstm_layer_forward(LSTM_layer *l, cl_mem x, cl_mem params, size_t t){
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
  check_error(clEnqueueNDRangeKernel(get_opencl_queue0(), logistic_kernel, 1, NULL, &l->size, NULL, 0, NULL, NULL), "gpu_lstm_layer_forward(): couldn't enqueue logistic kernel");

  check_error(clSetKernelArg(logistic_kernel, 0, sizeof(cl_mem), &l->input_gate_z[t]), "setting logistic arg 0");
  check_error(clSetKernelArg(logistic_kernel, 1, sizeof(cl_mem), &l->input_gate_output[t]), "setting logistic arg 0");
  check_error(clSetKernelArg(logistic_kernel, 2, sizeof(Nonlinearity), &gate_fn), "setting logistic arg 0");
  check_error(clEnqueueNDRangeKernel(get_opencl_queue0(), logistic_kernel, 1, NULL, &l->size, NULL, 0, NULL, NULL), "gpu_lstm_layer_forward(): couldn't enqueue logistic kernel");

  check_error(clSetKernelArg(logistic_kernel, 0, sizeof(cl_mem), &l->forget_gate_z[t]), "setting logistic arg 0");
  check_error(clSetKernelArg(logistic_kernel, 1, sizeof(cl_mem), &l->forget_gate_output[t]), "setting logistic arg 0");
  check_error(clSetKernelArg(logistic_kernel, 2, sizeof(Nonlinearity), &gate_fn), "setting logistic arg 0");
  check_error(clEnqueueNDRangeKernel(get_opencl_queue0(), logistic_kernel, 1, NULL, &l->size, NULL, 0, NULL, NULL), "couldn't enqueue logistic kernel");

  check_error(clSetKernelArg(logistic_kernel, 0, sizeof(cl_mem), &l->output_gate_z[t]), "setting logistic arg 0");
  check_error(clSetKernelArg(logistic_kernel, 1, sizeof(cl_mem), &l->output_gate_output[t]), "setting logistic arg 1");
  check_error(clSetKernelArg(logistic_kernel, 2, sizeof(Nonlinearity), &gate_fn), "setting logistic arg 2");
  check_error(clEnqueueNDRangeKernel(get_opencl_queue0(), logistic_kernel, 1, NULL, &l->size, NULL, 0, NULL, NULL), "couldn't enqueue logistic kernel");

  check_error(clSetKernelArg(lstm_forward_kernel, 0, sizeof(cl_mem), &l->input_nonl_output[t]), "setting lstm forward arg 0");
  check_error(clSetKernelArg(lstm_forward_kernel, 1, sizeof(cl_mem), &l->input_gate_output[t]), "setting lstm forward arg 1");
  check_error(clSetKernelArg(lstm_forward_kernel, 2, sizeof(cl_mem), &l->forget_gate_output[t]), "setting lstm forward arg 2");
  check_error(clSetKernelArg(lstm_forward_kernel, 3, sizeof(cl_mem), &l->output_gate_output[t]), "setting lstm forward arg 3");
  check_error(clSetKernelArg(lstm_forward_kernel, 4, sizeof(cl_mem), &l->cell_state[t]), "setting lstm forward arg 4");
  check_error(clSetKernelArg(lstm_forward_kernel, 5, sizeof(cl_mem), &l->lstate), "setting lstm forward arg 4");
  check_error(clSetKernelArg(lstm_forward_kernel, 6, sizeof(cl_mem), &l->output[t]), "setting lstm forward arg 4");
  check_error(clEnqueueNDRangeKernel(get_opencl_queue0(), lstm_forward_kernel, 1, NULL, &l->size, NULL, 0, NULL, NULL), "couldn't lstm forward kernel");

  check_error(clEnqueueCopyBuffer(get_opencl_queue0(), l->output[t], l->loutput, 0, 0, l->size * sizeof(float), 0, NULL, NULL), "copying output to loutput");
  check_error(clEnqueueCopyBuffer(get_opencl_queue0(), l->cell_state[t], l->lstate, 0, 0, l->size * sizeof(float), 0, NULL, NULL), "copying cell state to lcell_state");
  check_error(clFinish(get_opencl_queue0()), "waiting for queue 0 to finish executing (forward pass)");
}
#endif

/*
 * Does a forward pass through the network
 */
#ifndef SIEKNET_USE_GPU
void cpu_lstm_forward(LSTM *n, float *x){
  size_t t = n->t;
  for(int i = 0; i < n->input_dimension; i++){
    n->network_input[t][i] = x[i];
  }
  //Feedforward through all layers
  float *input = n->network_input[t];
  for(int i = 0; i < n->depth; i++){
    LSTM_layer *l = &n->layers[i];
    cpu_lstm_layer_forward(l, input, n->params, t);
    input = l->output[t];
  }
  cpu_mlp_layer_forward(&n->output_layer, input, n->params);
}
#else
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

  //Feedforward through all layers
  cl_mem input = n->network_input[t];
  for(int i = 0; i < n->depth; i++){
    LSTM_layer *l = &n->layers[i];
    gpu_lstm_layer_forward(l, input, n->params, t);
    input = l->output[t];
  }
  gpu_mlp_layer_forward(&n->output_layer, input, n->params);

#ifndef SIEKNET_GPU_NO_OUTPUT
  check_error(clEnqueueReadBuffer(get_opencl_queue0(), n->output_layer.output, 1, 0, sizeof(float) * n->output_layer.size, n->output, 0, NULL, NULL), "error reading output from gpu");

  check_error(clFinish(get_opencl_queue0()), "waiting for kernels to finish executing (forward pass)");
#endif
}
#endif

#ifndef SIEKNET_USE_GPU
float cpu_lstm_cost(LSTM *n, float *y){
  MLP_layer *mlp = &n->output_layer;
  float *o = mlp->output;
  float c = cpu_cost(o, y, n->cost_gradient, n->output_dimension, n->cost_fn);

  cpu_mlp_layer_backward(mlp, n->cost_gradient, n->params, n->param_grad);
  float *grads = mlp->input_gradient;

  /* copy gradient serially from mlp output layer to lstm network gradient. */
  for(int i = 0; i < mlp->input_dimension; i++)
    n->recurrent_gradient[n->t][i] = grads[i];

  return c;
}
#else
float gpu_lstm_cost(LSTM *n, float *y){
  MLP_layer *mlp = &n->output_layer;
  cl_mem o = mlp->output;
#ifdef SIEKNET_ONEHOT_SPEEDUP
  int m = argmax(y, n->output_dimension);
  check_error(clSetKernelArg(make_onehot_kernel, 0, sizeof(cl_mem), &n->output_label), "couldn't set onehot arg 0");
  check_error(clSetKernelArg(make_onehot_kernel, 1, sizeof(int), &m), "couldn't set onehot arg 1");
  check_error(clEnqueueNDRangeKernel(get_opencl_queue0(), make_onehot_kernel, 1, NULL, &n->output_dimension, NULL, 0, NULL, NULL), "couldn't do onehot kernel");
#else
  check_error(clEnqueueWriteBuffer(get_opencl_queue0(), n->output_label, 1, 0, sizeof(float) * n->output_dimension, y, 0, NULL, NULL), "enqueuing label");
#endif

  float c = gpu_cost(o, n->output_label, n->cost_gradient, n->output_dimension, n->cost_fn);

  gpu_mlp_layer_backward(mlp, n->cost_gradient, n->params, n->param_grad);

  check_error(clEnqueueCopyBuffer(get_opencl_queue0(), mlp->input_gradient, n->recurrent_gradient[n->t], 0, 0, sizeof(float) * mlp->input_dimension, 0, NULL, NULL), "copying mlp grads to lstm network grads");

  return c;

}
#endif

#ifndef SIEKNET_USE_GPU
void cpu_lstm_layer_backward(LSTM_layer *l, float **grads, float *params, float *param_grad, size_t MAX_TIME){

  int recurrent_offset = l->input_dimension - l->size;

  Nonlinearity gate_fn = sigmoid;
  Nonlinearity nonl_fn = hypertan;

  int params_per_cell = (l->input_dimension+1)*4;

  for(int t = MAX_TIME; t >= 0; t--){
    int use_future_grads, use_past_outputs;
    if(t >= MAX_TIME) use_future_grads = 0;
    else              use_future_grads = 1;

    if(!t) use_past_outputs = 0;
    else   use_past_outputs = 1;

    for(int i = 0; i < l->size; i++){
      if(use_future_grads){
        agnostic_lstm_dstate_kernel(grads[t],
                                    l->cell_state[t],
                                    l->output_gate_output[t],
                                    l->cell_dstate[t+1],
                                    l->forget_gate_output[t+1],
                                    l->input_gradient[t+1],
                                    l->cell_dstate[t],
                                    recurrent_offset,
                                    use_future_grads,
                                    i);
      }else{
        agnostic_lstm_dstate_kernel(grads[t],
                                    l->cell_state[t],
                                    l->output_gate_output[t],
                                    NULL,
                                    NULL,
                                    NULL,
                                    l->cell_dstate[t],
                                    recurrent_offset,
                                    use_future_grads,
                                    i);
      }
      agnostic_lstm_input_nonl_gradient_kernel(l->cell_dstate[t],
                                               l->input_gate_output[t],
                                               l->input_nonl_output[t],
                                               l->input_nonl_gradient[t],
                                               nonl_fn,
                                               i);
      agnostic_lstm_input_nonl_gradient_kernel(l->cell_dstate[t],
                                               l->input_nonl_output[t],
                                               l->input_gate_output[t],
                                               l->input_gate_gradient[t],
                                               gate_fn,
                                               i);

      if(use_past_outputs){
        agnostic_lstm_forget_gate_gradient_kernel(l->cell_dstate[t],
                                                  l->cell_state[t-1],
                                                  l->forget_gate_output[t],
                                                  l->forget_gate_gradient[t],
                                                  gate_fn,
                                                  use_past_outputs,
                                                  i);
      }else{
        agnostic_lstm_forget_gate_gradient_kernel(l->cell_dstate[t],
                                                  NULL,
                                                  l->forget_gate_output[t],
                                                  l->forget_gate_gradient[t],
                                                  gate_fn,
                                                  use_past_outputs,
                                                  i);
      }
      if(use_future_grads){
        agnostic_lstm_output_gate_gradient_kernel(grads[t],
            l->cell_state[t],
            l->output_gate_output[t],
            l->input_gradient[t+1],
            l->output_gate_gradient[t],
            gate_fn,
            recurrent_offset,
            use_future_grads,
            i);
        agnostic_lstm_parameter_gradient_kernel(l->input_nonl_gradient[t],
            l->input_gate_gradient[t],
            l->forget_gate_gradient[t],
            l->output_gate_gradient[t],
            l->input_nonl_gradient[t+1],
            l->input_gate_gradient[t+1],
            l->forget_gate_gradient[t+1],
            l->output_gate_gradient[t+1],
            param_grad,
            l->input[t],
            l->output[t],
            use_future_grads,
            l->size,
            l->input_dimension,
            l->param_offset,
            params_per_cell,
            i);

      }else{
        agnostic_lstm_output_gate_gradient_kernel(grads[t],
            l->cell_state[t],
            l->output_gate_output[t],
            NULL,
            l->output_gate_gradient[t],
            gate_fn,
            recurrent_offset,
            use_future_grads,
            i);
        agnostic_lstm_parameter_gradient_kernel(l->input_nonl_gradient[t],
            l->input_gate_gradient[t],
            l->forget_gate_gradient[t],
            l->output_gate_gradient[t],
            NULL,
            NULL,
            NULL,
            NULL,
            param_grad,
            l->input[t],
            l->output[t],
            use_future_grads,
            l->size,
            l->input_dimension,
            l->param_offset,
            params_per_cell,
            i);

      }
    }
    for(int i = 0; i < l->input_dimension; i++){
      agnostic_lstm_input_gradient_kernel(l->input_nonl_gradient[t],
          l->input_gate_gradient[t],
          l->forget_gate_gradient[t],
          l->output_gate_gradient[t],
          params,
          l->input_gradient[t],
          l->size,
          l->input_dimension,
          l->param_offset,
          params_per_cell,
          i);

    }
  }
}
#else
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
      check_error(clSetKernelArg(lstm_parameter_gradient_kernel, 4, sizeof(cl_mem), NULL), "lstm input grad arg 4");
      check_error(clSetKernelArg(lstm_parameter_gradient_kernel, 5, sizeof(cl_mem), NULL), "lstm input grad arg 5");
      check_error(clSetKernelArg(lstm_parameter_gradient_kernel, 6, sizeof(cl_mem), NULL), "lstm input grad arg 6");
      check_error(clSetKernelArg(lstm_parameter_gradient_kernel, 7, sizeof(cl_mem), NULL), "lstm input grad arg 7");
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

#endif

/*
 * Performs parameter gradient calculation for all LSTM layers
 */
#ifndef SIEKNET_USE_GPU
void cpu_lstm_backward(LSTM *n){
  if(n->t >= n->seq_len){
    float **grads = n->recurrent_gradient;
    for(int i = n->depth-1; i >= 0; i--){
      cpu_lstm_layer_backward(&n->layers[i], grads, n->params, n->param_grad, n->t-1);
      grads = n->layers[i].input_gradient;
    }
    if(!n->stateful) cpu_lstm_wipe(n);
    n->t = 0;
  }
}
#else
static void gpu_lstm_backward(LSTM *n){
  if(n->t >= n->seq_len){
    cl_mem *grads = n->recurrent_gradient;
    for(int i = n->depth-1; i >= 0; i--){
      LSTM_layer *l = &n->layers[i];
      gpu_lstm_layer_backward(l, grads, n->params, n->param_grad, n->t-1);
      grads = n->layers[i].input_gradient;
    }
    if(!n->stateful) gpu_lstm_wipe(n);
    n->t = 0;
    check_error(clFinish(get_opencl_queue0()), "waiting for kernels to finish executing (backward pass)");
  }
}
#endif

LSTM lstm_from_arr(size_t *arr, size_t len){
#ifdef SIEKNET_USE_GPU
  return gpu_lstm_from_arr(arr, len);
#else
  return cpu_lstm_from_arr(arr, len);
#endif
}

void lstm_wipe(LSTM *n){
#ifdef SIEKNET_USE_GPU
  gpu_lstm_wipe(n);
#else
  cpu_lstm_wipe(n);
#endif
}

void lstm_forward(LSTM *n, float *x){
#ifdef SIEKNET_USE_GPU
  gpu_lstm_forward(n, x);
#else
  cpu_lstm_forward(n, x);
#endif
  if(!(rand()%3))
    n->guess = sample_softmax(n->output, n->output_dimension);
  else
    n->guess = argmax(n->output, n->output_dimension);
}

/*
 * Calculates the cost gradient for an lstm given a label vector y.
 * y is expected to be of size n.output_dimension. 
 */
float lstm_cost(LSTM *n, float *y){
#ifndef SIEKNET_USE_GPU
  float c = cpu_lstm_cost(n, y);
#else
  float c = gpu_lstm_cost(n, y);
#endif
  n->t++;
  return c;
}

void lstm_backward(LSTM *n){
#ifdef SIEKNET_USE_GPU
  gpu_lstm_backward(n);
#else
  cpu_lstm_backward(n);
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
     for(int t = 0; t < SIEKNET_MAX_UNROLL_LENGTH; t++){
     free(l->input_gradient[t]);
     free(l->output[t]);
     }
     free(l->input_gradient);
     free(l->output);
     free(l->cells);
     }
     for(int t = 0; t < SIEKNET_MAX_UNROLL_LENGTH; t++){
     free(n->recurrent_gradient[t]);
     free(n->network_input[t]);
     }
     free(n->recurrent_gradient);
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

#ifdef SIEKNET_USE_GPU
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
#ifndef SIEKNET_USE_GPU
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
