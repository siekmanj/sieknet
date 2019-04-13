#include <rnn.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#ifdef SIEKNET_USE_GPU
#include <opencl_utils.h>
static cl_kernel make_onehot_kernel;
static cl_kernel rnn_forward_kernel;
static cl_kernel rnn_input_gradient_kernel, rnn_parameter_gradient_kernel;
static cl_kernel softmax_sum_kernel, softmax_kernel;
static cl_kernel cost_kernel, cost_gradient_kernel;
static cl_kernel logistic_kernel, zero_init_kernel;

static int ARE_KERNELS_INITIALIZED = 0;
void rnn_kernel_setup(){
  char *kernels[] = {"include/logistic.h", "include/mlp.h", "include/rnn.h", "src/logistic.cl", "src/mlp.cl", "src/rnn.cl"};

  int err = 0;

  char *src = get_kernel_source(kernels, sizeof(kernels)/sizeof(kernels[0]));
  cl_program prog = build_program(src);
  free(src);

  rnn_forward_kernel = clCreateKernel(prog, "rnn_forward_kernel", &err);
  check_error(err, "couldn't make forwards kernel");

  rnn_input_gradient_kernel = clCreateKernel(prog, "rnn_input_gradient_kernel", &err);
  check_error(err, "couldn't make mlp input grad kernel");

  rnn_parameter_gradient_kernel = clCreateKernel(prog, "rnn_parameter_gradient_kernel", &err);
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

  make_onehot_kernel = clCreateKernel(prog, "onehot_kernel", &err);
  check_error(err, "couldn't make onehot kernel");

  ARE_KERNELS_INITIALIZED = 1;
}
#endif

#ifndef SIEKNET_USE_GPU
void cpu_rnn_wipe(RNN *n){
  for(int i = 0; i < n->depth; i++){
    RNN_layer *l = &n->layers[i];
    //cpu_zero_2d_arr(&l->loutput, SIEKNET_MAX_UNROLL_LENGTH, 1);
    for(int j = 0; j < l->size; j++)
      l->loutput[j] = 0.0f;
  }
}
#else
void gpu_rnn_wipe(RNN *n){
  for(int i = 0; i < n->depth; i++){
    RNN_layer *l = &n->layers[i];
    check_error(clSetKernelArg(zero_init_kernel, 0, sizeof(cl_mem), &l->loutput), "couldn't set zero kernel arg 0");
    check_error(clEnqueueNDRangeKernel(get_opencl_queue0(), zero_init_kernel, 1, NULL, &l->size, NULL, 0, NULL, NULL), "couldn't use zero kernel");
  }
}
#endif

#ifndef SIEKNET_USE_GPU
RNN_layer cpu_create_RNN_layer(size_t input_dim, size_t size, float *params, int param_offset, Nonlinearity logistic){
  RNN_layer l;
  l.size = size;
  l.logistic = logistic;
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

  l.input   = ALLOC(float *, SIEKNET_MAX_UNROLL_LENGTH);
  l.loutput = ALLOC(float, l.size);

  cpu_zero_2d_arr(l.input_gradient, SIEKNET_MAX_UNROLL_LENGTH, l.input_dimension);
  return l;
}
#else
RNN_layer gpu_create_RNN_layer(size_t input_dim, size_t size, cl_mem params, int param_offset, Nonlinearity logistic){
  RNN_layer l;
  l.size = size;
  l.logistic = logistic;
  l.param_offset = param_offset;
  l.input_dimension = input_dim + size;

  size_t neuron_offset = param_offset;
  float *tmp = (float*)malloc(sizeof(float) * l.size * (l.input_dimension+1));
  for(int i = 0; i < size; i++){
    xavier_init(&tmp[neuron_offset], (l.input_dimension+1), l.size);
    neuron_offset += l.input_dimension+1;
  }
  check_error(clEnqueueWriteBuffer(get_opencl_queue0(), params, 1, sizeof(float) * param_offset, sizeof(float)*l.size*(l.input_dimension+1), tmp, 0, NULL, NULL), "could not enqueue layer params");
  free(tmp);

  l.z = ALLOC(cl_mem, SIEKNET_MAX_UNROLL_LENGTH);
  l.output = ALLOC(cl_mem, SIEKNET_MAX_UNROLL_LENGTH);
  l.input_gradient = ALLOC(cl_mem, SIEKNET_MAX_UNROLL_LENGTH);
  return l;
}

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
  printf("num params: %lu\n", num_params);

  n.layers = ALLOC(RNN_layer, len-2);
  n.params = ALLOC(float, num_params);
  n.param_grad = ALLOC(float, num_params);

  memset(n.param_grad, '\0', num_params*sizeof(float));

  int param_idx = 0;
  for(int i = 1; i < len-1; i++){
    printf("giving layer %d param offset of %d\n", i-1, param_idx);
    RNN_layer l = cpu_create_RNN_layer(arr[i-1], arr[i], n.params, param_idx, hypertan);
    param_idx += (arr[i-1]+arr[i]+1)*arr[i];
    n.layers[i-1] = l;
  }
  n.output_layer = cpu_create_MLP_layer(arr[len-2], arr[len-1], n.params, param_idx, softmax);

  n.recurrent_gradient = alloc_2d_array(SIEKNET_MAX_UNROLL_LENGTH, arr[len-2]);
  n.network_input      = alloc_2d_array(SIEKNET_MAX_UNROLL_LENGTH, arr[0]);

  n.cost_gradient = ALLOC(float, n.output_dimension);

  cpu_zero_2d_arr(n.recurrent_gradient, SIEKNET_MAX_UNROLL_LENGTH, arr[len-2]);
  cpu_zero_2d_arr(n.network_input, SIEKNET_MAX_UNROLL_LENGTH, arr[0]);

  n.output = n.output_layer.output;
  getchar();
  return n;
}
#else
RNN gpu_rnn_from_arr(const size_t *arr, const size_t len){
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
    RNN_layer l = gpu_create_RNN_layer(arr[i-1], arr[i], n.params, param_idx, sigmoid);
    param_idx += ((arr[i-1]+arr[i]+1))*arr[i];
    n.layers[i-1] = l;
  }
  n.output_layer = gpu_create_MLP_layer(arr[len-2], arr[len-1], n.params, param_idx, softmax);
  n.output = ALLOC(float, n.output_dimension);

  gpu_rnn_wipe(&n);
  return n;
}
#endif

#ifndef SIEKNET_USE_GPU
void cpu_rnn_layer_forward(RNN_layer *l, float *x, const float *params, size_t t){
  l->input[t] = x;
  int params_per_neuron = l->input_dimension+1;
  for(int i = 0; i < l->size; i++){
    l->z[t][i] = 0;
    agnostic_rnn_forward_kernel(x, l->loutput, l->z[t], params, l->input_dimension, l->size, l->param_offset, params_per_neuron, i);
    l->output[t][i] = activate(l->z[t][i], l->logistic);
  }
  for(int i = 0; i < l->size; i++)
    l->loutput[i] = l->output[t][i];
}
#else
static void gpu_rnn_layer_forward(RNN_layer *l, cl_mem x, cl_mem params, size_t t){
  l->input[t] = x;
  int params_per_neuron = l->input_dimension+1;

  check_error(clSetKernelArg(rnn_forward_kernel, 0, sizeof(cl_mem), &x), "setting forward kernel arg0");
  check_error(clSetKernelArg(rnn_forward_kernel, 1, sizeof(cl_mem), &l->loutput), "setting forward kernel arg1");
  check_error(clSetKernelArg(rnn_forward_kernel, 2, sizeof(cl_mem), &l->z[t]), "setting forward kernel arg2");
  check_error(clSetKernelArg(rnn_forward_kernel, 3, sizeof(cl_mem), &params), "setting forward kernel arg3");
  check_error(clSetKernelArg(rnn_forward_kernel, 4, sizeof(int), &l->input_dimension), "setting forward kernel arg4");
  check_error(clSetKernelArg(rnn_forward_kernel, 5, sizeof(int), &l->size), "setting forward kernel arg4");
  check_error(clSetKernelArg(rnn_forward_kernel, 6, sizeof(int), &l->param_offset), "setting forward kernel arg5");
  check_error(clSetKernelArg(rnn_forward_kernel, 7, sizeof(int), &params_per_neuron), "setting forward kernel arg6");
  check_error(clEnqueueNDRangeKernel(get_opencl_queue0(), rnn_forward_kernel, 1, NULL, &l->size, NULL, 0, NULL, NULL), "gpu_lstm_layer_forward(): couldn't enqueue linear kernel");

  check_error(clSetKernelArg(logistic_kernel, 0, sizeof(cl_mem), &l->z[t]), "setting logistic arg 0");
  check_error(clSetKernelArg(logistic_kernel, 1, sizeof(cl_mem), &l->output[t]), "setting logistic arg 1");
  check_error(clSetKernelArg(logistic_kernel, 2, sizeof(Nonlinearity), &l->logistic), "setting logistic arg 2");
  check_error(clEnqueueNDRangeKernel(get_opencl_queue0(), logistic_kernel, 1, NULL, &l->size, NULL, 0, NULL, NULL), "couldn't enqueue logistic kernel");

  check_error(clEnqueueCopyBuffer(get_opencl_queue0(), l->output[t], l->loutput, 0, 0, l->size * sizeof(float), 0, NULL, NULL), "copying output to loutput");
  check_error(clFinish(get_opencl_queue0()), "waiting for queue 0 to finish executing (forward pass)");

}
#endif

#ifndef SIEKNET_USE_GPU
void cpu_rnn_forward(RNN *n, const float *x){
  size_t t = n->t;
  for(int i = 0; i < n->input_dimension; i++)
    n->network_input[t][i] = x[i];

  float *input = n->network_input[t];
  for(int i = 0; i < n->depth; i++){
    RNN_layer *l = &n->layers[i];
    cpu_rnn_layer_forward(l, input, n->params, t);
    input = l->output[t];
  }
  cpu_mlp_layer_forward(&n->output_layer, input, n->params);

}
#else
void gpu_rnn_forward(RNN *n, const float *x){
  size_t t = n->t;
  check_error(clEnqueueWriteBuffer(get_opencl_queue0(), n->network_input[t], 1, 0, sizeof(float) * n->input_dimension, x, 0, NULL, NULL), "copying input");

  cl_mem input = n->network_input[t];
  for(int i = 0; i < n->depth; i++){
    RNN_layer *l = &n->layers[i];
    gpu_rnn_layer_forward(l, input, n->params, t);
    input = l->output[t];
  }
  gpu_mlp_layer_forward(&n->output_layer, input, n->params);

  check_error(clEnqueueReadBuffer(get_opencl_queue0(), n->output_layer.output, 1, 0, sizeof(float) * n->output_layer.size, n->output, 0, NULL, NULL), "error reading output from gpu");

  check_error(clFinish(get_opencl_queue0()), "waiting for kernels to finish executing (forward pass)");
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
  MLP_layer *mlp = &n->output_layer;
  check_error(clEnqueueWriteBuffer(get_opencl_queue0(), n->output_label, 1, 0, sizeof(float) * n->output_dimension, y, 0, NULL, NULL), "enqueuing label");
  cl_mem o = mlp->output;
  cl_mem dest = n->cost_gradient;

  float c = gpu_cost(o, n->output_label, dest, n->output_dimension, n->cost_fn);

  gpu_mlp_layer_backward(mlp, n->cost_gradient, n->params, n->param_grad);
  
  check_error(clEnqueueCopyBuffer(get_opencl_queue0(), mlp->input_gradient, n->recurrent_gradient[n->t], 0, 0, sizeof(float) * mlp->input_dimension, 0, NULL, NULL), "copying mlp input grad to rnn grad");

  return c;
}
#endif

#ifndef SIEKNET_USE_GPU
void cpu_rnn_layer_backward(RNN_layer *l, float **grad, float *params, float *param_grad, size_t MAX_TIME){
  int params_per_neuron = (l->input_dimension+1);

  for(int t = MAX_TIME; t >= 0; t--){
    int use_future_grads, use_past_outputs;
    if(t >= MAX_TIME) use_future_grads = 0;
    else              use_future_grads = 1;

    if(!t) use_past_outputs = 0;
    else   use_past_outputs = 1;

    float *future_input_gradient, *previous_output;

    if(use_future_grads)
      future_input_gradient = l->input_gradient[t+1];
    else
      future_input_gradient = NULL;

    if(use_past_outputs)
      previous_output = l->output[t-1];
    else
      previous_output = NULL;

    for(int i = 0; i < l->input_dimension; i++){
      agnostic_rnn_input_gradient_kernel(grad[t],
                                         l->output[t],
                                         params,
                                         future_input_gradient,
                                         l->input_gradient[t],
                                         l->logistic,
                                         use_future_grads,
                                         l->input_dimension,
                                         l->size,
                                         l->param_offset,
                                         params_per_neuron,
                                         i);
    }
    for(int i = 0; i < l->size; i++){
      agnostic_rnn_parameter_gradient_kernel(grad[t],
                                             l->output[t],
                                             future_input_gradient,
                                             previous_output,
                                             l->input[t],
                                             param_grad,
                                             l->logistic,
                                             use_future_grads,
                                             use_past_outputs,
                                             l->input_dimension,
                                             l->size,
                                             l->param_offset,
                                             params_per_neuron,
                                             i);
    }
  }
}
#else
void gpu_rnn_layer_backward(RNN_layer *l, cl_mem grad, cl_mem params, cl_mem param_grad){

}
#endif

#ifndef SIEKNET_USE_GPU
void cpu_rnn_backward(RNN *n){
  if(n->t >= n->seq_len){
    float **grads = n->recurrent_gradient;
    for(int i = n->depth-1; i >= 0; i--){
      RNN_layer *l = &n->layers[i];
      cpu_rnn_layer_backward(l, grads, n->params, n->param_grad, n->t-1);
      grads = l->input_gradient;
    }
    if(!n->stateful) cpu_rnn_wipe(n);
    n->t = 0;
  }
}
#else
void gpu_rnn_backward(RNN *n){
  if(n->t >= n->seq_len){
    cl_mem *grads = n->recurrent_gradient;
    for(int i = n->depth-1; i >= 0; i--){
      RNN_layer *l = &n->layers[i];
      gpu_rnn_layer_backward(l, grads, n->params, n->param_grad, n->t-1);
      grads = l->input_gradient;
    }
    if(!n->stateful) gpu_rnn_wipe(n);
    n->t = 0;
  }

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
  if(!(rand()%3))
    n->guess = sample_softmax(n->output, n->output_dimension);
  else
    n->guess = argmax(n->output, n->output_dimension);
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

void rnn_wipe(RNN *n){
#ifndef SIEKNET_USE_GPU
  cpu_rnn_wipe(n);
#else
  gpu_rnn_wipe(n);
#endif
}




