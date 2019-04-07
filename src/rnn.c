#include <rnn.h>
#include <stdio.h>
#include <string.h>

#ifdef SIEKNET_USE_GPU
static cl_kernel make_onehot_kernel;
static cl_kernel rnn_forward_kernel;
static cl_kernel rnn_input_gradient_kernel, rnn_parameter_gradient_kernel;
static cl_kernel softmax_sum_kernel, softmax_kernel;
static cl_kernel logistic_kernel, zero_init_kernel;

static int ARE_KERNELS_INITIALIZED = 0;
void rnn_kernel_setup(){
  char *kernels[] = {"include/logistic.h", "include/mlp.h", "include/rnn.h", "src/logistic.cl", "src/mlp.cl", "src/rnn.cl"};

  int err = 0;

  char *src = get_kernel_source(kernels, 6);
  cl_program prog = build_program(src);
  free(src);

  mlp_forward_kernel = clCreateKernel(prog, "rnn_forward_kernel", &err);
  check_error(err, "couldn't make forwards kernel");

  mlp_input_gradient_kernel = clCreateKernel(prog, "rnn_input_gradient_kernel", &err);
  check_error(err, "couldn't make mlp input grad kernel");

  mlp_parameter_gradient_kernel = clCreateKernel(prog, "rnn_parameter_gradient_kernel", &err);
  check_error(err, "couldn't make mlp param grad kernel");

  logistic_kernel = clCreateKernel(prog, "logistic_kernel", &err);
  check_error(err, "couldn't make sigmoid kernel");

  zero_init_kernel = clCreateKernel(prog, "zero_init_kernel", &err);
  check_error(err, "couldn't make zero init kernel");

  softmax_sum_kernel = clCreateKernel(prog, "softmax_sum_kernel", &err);
  check_error(err, "couldn't make softmax sum kernel");

  softmax_kernel = clCreateKernel(prog, "softmax_kernel", &err);
  check_error(err, "couldn't make softmax kernel");

  cost_kernel = clCreateKernel(prog, "cost_kernel", &err);
  check_error(err, "couldn't make scalar cost kernel");

  cost_gradient_kernel = clCreateKernel(prog, "cost_gradient_kernel", &err);
  check_error(err, "couldn't make cost gradient kernel");

  ARE_KERNELS_INITIALIZED = 1;
}
#endif

#ifndef SIEKNET_USE_GPU
RNN_layer cpu_create_RNN_layer(size_t input_dim, size_t size, float *params, int param_offset){
  RNN_layer l;
  l.size = size;
  l.param_offset = param_offset;
  l.input_dimension = input_dim + size;

  size_t neuron_offset = param_offset;
  for(int i = 0; i < size; i++){
    xavier_init(&params[neuron_offset], (l.input_dimension+1), l.size);
    neuron_offset += l.input_dimension+1;
  }

  l.z = alloc_2d_array(SIEKNET_MAX_UNROLL_LENGTH, l.size);
  l.output = alloc_2d_array(SIEKNET_MAX_UNROLL_LENGTH, l.size);
  l.input_gradient = alloc_2d_array(SIEKNET_MAX_UNROLL_LENGTH, l.input_dimension);

  l.input = ALLOC(float *, SIEKNET_MAX_UNROLL_LENGTH);
  l.loutput = ALLOC(float, l.size);

  cpu_zero_2d_arr(l.input_gradient, SIEKNET_MAX_UNROLL_LENGTH, l.input_dimension);
  return l;
}
#else

#endif

#ifndef SIEKNET_USE_GPU
RNN cpu_rnn_from_arr(const size_t *arr, const size_t len){
  RNN n;
  n.t = 0;
  n.stateful = 0;
  n.seq_len = 25;
  n.input_dimension = arr[0];
  n.output_dimension = arr[len-1];
  n.depth = len-2;
  n.cost_fn = cross_entropy;

  if(len < 3){
    printf("ERROR: rnn_from_arr(): must have at least one input dim, hidden layer size, and output dim (3 layers) but only %lu layers provided.\n", len);
    exit(1);
  }

  size_t num_params = 0;
  for(int i = 1; i < len-1; i++)
    num_params += ((arr[i-1]+arr[i]+1)*arr[i]); //parameters for input, recurrent input, and bias term
  num_params += (arr[len-2]+1)*arr[len-1]; //output mlp layer params
  n.num_params = num_params;

  n.layers = ALLOC(RNN_layer, len-2);
  n.params = ALLOC(float, num_params);
  n.param_grad = ALLOC(float, num_params);

  memset(n.param_grad, '\0', num_params*sizeof(float));

  int param_idx = 0;
  for(int i = 1; i < len-1; i++){
    RNN_layer l = cpu_create_RNN_layer(arr[i-1], arr[i], n.params, param_idx);
    param_idx += (arr[i-1]+arr[i]+1)*arr[i];
    n.layers[i-1] = l;
  }

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
RNN gpu_rnn_from_arr(const size_t *arr, const size_t depth){

}
#endif

#ifndef SIEKNET_USE_GPU
void cpu_rnn_forward(RNN *n, const float *x){

}
#else
void gpu_rnn_forward(RNN *n, const float *x){

}
#endif

#ifndef SIEKNET_USE_GPU
float cpu_rnn_cost(RNN *n, const float *y){
  MLP_layer *mlp = &n->output_layer;
  float *o = mlp->output;
  float *dest = n->cost_gradient;
  float c = cpu_cost(o, y, dest, n->output_dimension, n->cost_fn);

  cpu_mlp_layer_backward(mlp, n->cost_gradient, n->params, n->param_grad);
  float *grads = mlp->input_gradient;

  for(int i = 0; i < mlp->input_dimension; i++)
    n->recurrent_gradient[n->t][i] = grads[i];

  return c;
}
#else
float gpu_rnn_cost(RNN *n, const float *y){
  check_error(clEnqueueWriteBuffer(get_opencl_queue0(), n->output_label, 1, 0, sizeof(float) * n->output_dimension, y, 0, NULL, NULL), "enqueuing label");
  cl_mem o = n->output_layer.output;
  cl_mem dest = n->cost_gradient;

  float c = gpu_cost(o, n->output_label, dest, n->output_dimension, n->cost_fn);

  gpu_mlp_layer_backward(mlp, n->cost_gradient, n->params, n->param_grad);
  
  check_error(clEnqueueCopyBuffer(get_opencl_queue0(), mlp->input_gradient, n->recurrent_gradient[n->t], 0, 0, sizeof(float) * mlp->input_dimension, 0, NULL, NULL), "copying mlp input grad to rnn grad");

  return c;
}
#endif

#ifndef SIEKNET_USE_GPU
void cpu_rnn_backward(RNN *n){

}
#else
void gpu_rnn_backward(RNN *n){

}
#endif

RNN rnn_from_arr(const size_t *arr, const size_t depth){
#ifndef SIEKNET_USE_GPU
  return cpu_rnn_from_arr(arr, depth);
#else
  return gpu_rnn_from_arr(arr, depth);
#endif
}

void rnn_forward(RNN *n, const float *x){
#ifndef SIEKNET_USE_GPU
  cpu_rnn_forward(n, x);
#else
  gpu_rnn_forward(n, x);
#endif
}

float rnn_cost(RNN *n, const float *y){
#ifndef SIEKNET_USE_GPU
  float c = cpu_rnn_cost(n, y);
#else
  float c = gpu_rnn_cost(n, y);
#endif
  n->t++;
  return c;
}

void rnn_backward(RNN *n){
#ifndef SIEKNET_USE_GPU
  cpu_rnn_backward(n);
#else
  gpu_rnn_backward(n);
#endif
}




