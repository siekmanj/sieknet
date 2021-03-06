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
  mlp_kernel_setup();
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

  make_onehot_kernel = clCreateKernel(prog, "make_onehot_kernel", &err);
  check_error(err, "couldn't make onehot kernel");

  ARE_KERNELS_INITIALIZED = 1;
}
#endif

#ifndef SIEKNET_USE_GPU
void cpu_rnn_wipe(RNN *n){
  for(int i = 0; i < n->depth; i++){
    RNN_layer *l = &n->layers[i];
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
  l.input = ALLOC(cl_mem, SIEKNET_MAX_UNROLL_LENGTH);
  l.output = ALLOC(cl_mem, SIEKNET_MAX_UNROLL_LENGTH);
  l.input_gradient = ALLOC(cl_mem, SIEKNET_MAX_UNROLL_LENGTH);

  int err;
  for(int t = 0; t < SIEKNET_MAX_UNROLL_LENGTH; t++){
    l.z[t]  = clCreateBuffer(get_opencl_context(), CL_MEM_READ_WRITE, sizeof(float) * l.size, NULL, &err); 
    check_error(err, "allocating internal rnn memory");
    l.output[t]  = clCreateBuffer(get_opencl_context(), CL_MEM_READ_WRITE, sizeof(float) * l.size, NULL, &err); 
    check_error(err, "allocating internal rnn memory");
    l.input_gradient[t]  = clCreateBuffer(get_opencl_context(), CL_MEM_READ_WRITE, sizeof(float) * l.input_dimension, NULL, &err); 
    check_error(err, "allocating internal rnn memory");
  }
  l.loutput = clCreateBuffer(get_opencl_context(), CL_MEM_READ_WRITE, sizeof(float) * l.size, NULL, &err); 
  check_error(err, "allocating internal rnn memory");
  gpu_zero_2d_arr(&l.loutput, 1, l.size);

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
  for(int i = 1; i < len-1; i++){
    num_params += ((arr[i-1]+arr[i]+1)*arr[i]); //parameters for input, recurrent input, and bias term
  }
  num_params += (arr[len-2]+1)*arr[len-1]; //output mlp layer params
  n.num_params = num_params;

  n.layers = ALLOC(RNN_layer, len-2);
  n.params = ALLOC(float, num_params);
  n.param_grad = ALLOC(float, num_params);

  memset(n.param_grad, '\0', num_params*sizeof(float));

  int param_idx = 0;
  for(int i = 1; i < len-1; i++){
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
  return n;
}
#else
RNN gpu_rnn_from_arr(const size_t *arr, const size_t len){
  initialize_opencl();
  if(!ARE_KERNELS_INITIALIZED)
    rnn_kernel_setup();
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
    RNN_layer l = gpu_create_RNN_layer(arr[i-1], arr[i], n.params, param_idx, hypertan);
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
  return c;
}
#else
float gpu_rnn_cost(RNN *n, const float *y){
  MLP_layer *mlp = &n->output_layer;
  check_error(clEnqueueWriteBuffer(get_opencl_queue0(), n->output_label, 1, 0, sizeof(float) * n->output_dimension, y, 0, NULL, NULL), "enqueuing label");
  cl_mem o = mlp->output;
  cl_mem dest = n->cost_gradient;

  float c = gpu_cost(o, n->output_label, dest, n->output_dimension, n->cost_fn);
  return c;
}
#endif

#ifndef SIEKNET_USE_GPU
void cpu_rnn_layer_backward(RNN_layer *l, float **grad, float *params, float *param_grad, int abs_grad, size_t MAX_TIME){
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
																						 abs_grad,
                                             i);
    }
  }
}
#else
void gpu_rnn_layer_backward(RNN_layer *l, cl_mem *grad, cl_mem params, cl_mem param_grad, int abs_grad, size_t MAX_TIME){
  int params_per_neuron = (l->input_dimension+1);

  for(int t = MAX_TIME; t >= 0; t--){
    int use_future_grads, use_past_outputs;
    if(t >= MAX_TIME) use_future_grads = 0;
    else              use_future_grads = 1;

    if(!t) use_past_outputs = 0;
    else   use_past_outputs = 1;

    cl_mem future_input_gradient, previous_output;

    if(use_future_grads)
      future_input_gradient = l->input_gradient[t+1];
    else
      future_input_gradient = NULL;

    if(use_past_outputs)
      previous_output = l->output[t-1];
    else
      previous_output = NULL;

    check_error(clSetKernelArg(rnn_input_gradient_kernel, 0,  sizeof(cl_mem), &grad[t]), "setting forward kernel arg0");
    check_error(clSetKernelArg(rnn_input_gradient_kernel, 1,  sizeof(cl_mem), &l->output[t]), "setting forward kernel arg1");
    check_error(clSetKernelArg(rnn_input_gradient_kernel, 2,  sizeof(cl_mem), &params), "setting forward kernel arg2");
    check_error(clSetKernelArg(rnn_input_gradient_kernel, 3,  sizeof(cl_mem), &future_input_gradient), "setting forward kernel arg3");
    check_error(clSetKernelArg(rnn_input_gradient_kernel, 4,  sizeof(cl_mem), &l->input_gradient[t]), "setting forward kernel arg4");
    check_error(clSetKernelArg(rnn_input_gradient_kernel, 5,  sizeof(Nonlinearity), &l->logistic), "setting forward kernel arg5");
    check_error(clSetKernelArg(rnn_input_gradient_kernel, 6,  sizeof(int), &use_future_grads), "setting forward kernel arg6");
    check_error(clSetKernelArg(rnn_input_gradient_kernel, 7,  sizeof(int), &l->input_dimension), "setting forward kernel arg7");
    check_error(clSetKernelArg(rnn_input_gradient_kernel, 8,  sizeof(int), &l->size), "setting forward kernel arg8");
    check_error(clSetKernelArg(rnn_input_gradient_kernel, 9,  sizeof(int), &l->param_offset), "setting forward kernel arg9");
    check_error(clSetKernelArg(rnn_input_gradient_kernel, 10, sizeof(int), &params_per_neuron), "setting forward kernel arg10");
    check_error(clEnqueueNDRangeKernel(get_opencl_queue0(), rnn_input_gradient_kernel, 1, NULL, &l->input_dimension, NULL, 0, NULL, NULL), "gpu_rnn_layer_backward(): couldn't enqueue input grad kernel");

    check_error(clSetKernelArg(rnn_parameter_gradient_kernel, 0,  sizeof(cl_mem), &grad[t]), "setting forward kernel arg0");
    check_error(clSetKernelArg(rnn_parameter_gradient_kernel, 1,  sizeof(cl_mem), &l->output[t]), "setting forward kernel arg1");
    check_error(clSetKernelArg(rnn_parameter_gradient_kernel, 2,  sizeof(cl_mem), &future_input_gradient), "setting forward kernel arg2");
    check_error(clSetKernelArg(rnn_parameter_gradient_kernel, 3,  sizeof(cl_mem), &previous_output), "setting forward kernel arg3");
    check_error(clSetKernelArg(rnn_parameter_gradient_kernel, 4,  sizeof(cl_mem), &l->input[t]), "setting forward kernel arg4");
    check_error(clSetKernelArg(rnn_parameter_gradient_kernel, 5,  sizeof(cl_mem), &param_grad), "setting forward kernel arg5");
    check_error(clSetKernelArg(rnn_parameter_gradient_kernel, 6,  sizeof(Nonlinearity), &l->logistic), "setting forward kernel arg6");
    check_error(clSetKernelArg(rnn_parameter_gradient_kernel, 7,  sizeof(int), &use_future_grads), "setting forward kernel arg7");
    check_error(clSetKernelArg(rnn_parameter_gradient_kernel, 8,  sizeof(int), &use_past_outputs), "setting forward kernel arg8");
    check_error(clSetKernelArg(rnn_parameter_gradient_kernel, 9,  sizeof(int), &l->input_dimension), "setting forward kernel arg9");
    check_error(clSetKernelArg(rnn_parameter_gradient_kernel, 10, sizeof(int), &l->size), "setting forward kernel arg10");
    check_error(clSetKernelArg(rnn_parameter_gradient_kernel, 11, sizeof(int), &l->param_offset), "setting forward kernel arg11");
    check_error(clSetKernelArg(rnn_parameter_gradient_kernel, 12, sizeof(int), &params_per_neuron), "setting forward kernel arg12");
    check_error(clSetKernelArg(rnn_parameter_gradient_kernel, 13, sizeof(int), &abs_grad), "setting forward kernel arg12");
    check_error(clEnqueueNDRangeKernel(get_opencl_queue0(), rnn_parameter_gradient_kernel, 1, NULL, &l->size, NULL, 0, NULL, NULL), "gpu_rnn_layer_backward(): couldn't enqueue param grad kernel");
  }

}
#endif

#ifndef SIEKNET_USE_GPU
void cpu_rnn_backward(RNN *n, int abs_grad){
	{
		MLP_layer *mlp = &n->output_layer;
		cpu_mlp_layer_backward(mlp, n->cost_gradient, n->params, n->param_grad, 0);
		float *grads = mlp->input_gradient;

		for(int i = 0; i < mlp->input_dimension; i++)
			n->recurrent_gradient[n->t][i] = grads[i];

		n->t++;
	}
  if(n->t >= n->seq_len){
    float **grads = n->recurrent_gradient;
    for(int i = n->depth-1; i >= 0; i--){
      RNN_layer *l = &n->layers[i];
      cpu_rnn_layer_backward(l, grads, n->params, n->param_grad, abs_grad, n->t-1);
      grads = l->input_gradient;
    }
    if(!n->stateful) cpu_rnn_wipe(n);
    n->t = 0;
  }
}
#else
void gpu_rnn_backward(RNN *n, int abs_grad){
	{
		MLP_layer *mlp = &n->output_layer;
		gpu_mlp_layer_backward(mlp, n->cost_gradient, n->params, n->param_grad, 0);
		check_error(clEnqueueCopyBuffer(get_opencl_queue0(), mlp->input_gradient, n->recurrent_gradient[n->t], 0, 0, sizeof(float) * mlp->input_dimension, 0, NULL, NULL), "copying mlp input grad to rnn grad");

		n->t++;
	}
  if(n->t >= n->seq_len){
    cl_mem *grads = n->recurrent_gradient;
    for(int i = n->depth-1; i >= 0; i--){
      RNN_layer *l = &n->layers[i];
      gpu_rnn_layer_backward(l, grads, n->params, n->param_grad, abs_grad, n->t-1);
      grads = l->input_gradient;
    }
    if(!n->stateful) gpu_rnn_wipe(n);
    n->t = 0;
    check_error(clFinish(get_opencl_queue0()), "waiting for kernels to finish executing (backward pass)");
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
  return c;
}

void rnn_backward(RNN *n){
#ifndef SIEKNET_USE_GPU
  cpu_rnn_backward(n, 0);
#else
  gpu_rnn_backward(n, 0);
#endif
}

void rnn_abs_backward(RNN *n){
#ifndef SIEKNET_USE_GPU
  cpu_rnn_backward(n, 1);
#else
  gpu_rnn_backward(n, 1);
#endif
}

void rnn_wipe(RNN *n){
#ifndef SIEKNET_USE_GPU
  cpu_rnn_wipe(n);
#else
  gpu_rnn_wipe(n);
#endif
}

/*
 * Does a deep-copy of an RNN.
 */
RNN *copy_rnn(RNN *n){
#ifndef SIEKNET_USE_GPU
	size_t arr[n->depth+2];
	arr[0] = n->input_dimension;
	for(int i = 0; i < n->depth; i++){
		arr[i+1] = n->layers[i].size;
	}
	arr[n->depth+1] = n->output_layer.size;

	RNN *ret = ALLOC(RNN, 1);
	*ret = rnn_from_arr(arr, n->depth+2);

	for(int i = 0; i < n->depth; i++){
		ret->layers[i].logistic = n->layers[i].logistic;
	}
	ret->output_layer.logistic = n->output_layer.logistic;

	for(int i = 0; i < n->num_params; i++)
		ret->params[i] = n->params[i];
  ret->performance = 0;

  return ret;
#else
  printf("WARNING: copy_rnn(): copying currently not supported on GPU\n");
  exit(1);
  return NULL;
#endif
}

void dealloc_rnn(RNN *n){
#ifndef SIEKNET_USE_GPU
  free(n->params);
  free(n->param_grad);
  for(int i = 0; i < n->depth; i++){
    RNN_layer *l = &n->layers[i];
    free(l->input);
    free(l->loutput);
    for(int t = 0; t < SIEKNET_MAX_UNROLL_LENGTH; t++){
      free(l->z[t]);
      free(l->output[t]);
      free(l->input_gradient[t]);
    }
    free(l->z);
    free(l->output);
    free(l->input_gradient);
  }
  for(int t = 0; t < SIEKNET_MAX_UNROLL_LENGTH; t++){
    free(n->recurrent_gradient[t]);
    free(n->network_input[t]);
  }
  free(n->recurrent_gradient);
  free(n->network_input);
  free(n->output_layer.z);
  free(n->output_layer.output);
  free(n->output_layer.input_gradient);
  free(n->cost_gradient);
  free(n->layers);
#else
  clReleaseMemObject(n->params);
  clReleaseMemObject(n->param_grad);
  for(int i = 0; i < n->depth; i++){
    RNN_layer *l = &n->layers[i];
    for(int t = 0; t < SIEKNET_MAX_UNROLL_LENGTH; t++){
      clReleaseMemObject(l->z[t]);
      clReleaseMemObject(l->output[t]);
      clReleaseMemObject(l->input_gradient[t]);
    }
    clReleaseMemObject(l->loutput);
    free(l->input);
    free(l->z);
    free(l->output);
    free(l->input_gradient);
  }
  for(int t = 0; t < SIEKNET_MAX_UNROLL_LENGTH; t++){
    clReleaseMemObject(n->recurrent_gradient[t]);
    clReleaseMemObject(n->network_input[t]);
  }
  clReleaseMemObject(n->cost_gradient);
  free(n->recurrent_gradient);
  free(n->network_input);
  free(n->layers);
  free(n->output);
  clReleaseMemObject(n->output_layer.z);
  clReleaseMemObject(n->output_layer.output);
  clReleaseMemObject(n->output_layer.input_gradient);
#endif
}

/*
 * IO FUNCTIONS FOR READING AND WRITING TO A FILE
 */

static int getWord(FILE *fp, char* dest){
  memset(dest, '\0', strlen(dest));
  return fscanf(fp, " %1023s", dest);
}

void save_rnn(RNN *n, const char *filename){
  FILE *fp = fopen(filename, "w");
  if(!fp){
    printf("ERROR: save_lstm(): could not open file '%s' (correct filepath? does dir exist?)", filename);
    exit(1);
  }
  fprintf(fp, "RNN %lu %lu ", n->depth, n->input_dimension);
  for(int i = 0; i < n->depth; i++){
    fprintf(fp, "%lu %u ", n->layers[i].size, n->layers[i].logistic);
  }
  fprintf(fp, "%lu %u\n", n->output_layer.size, n->output_layer.logistic);

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

RNN load_rnn(const char *filename){
  FILE *fp = fopen(filename, "rb");
  char buff[1024];
  memset(buff, '\0', 1024);

  getWord(fp, buff); //Get first word to check if MLP file

  if(strcmp(buff, "RNN") != 0){
    printf("ERROR: [%s] is not an RNN.\n", buff);
    exit(1);
  }
  size_t num_layers, input_dim;
  if(fscanf(fp, "%lu %lu", &num_layers, &input_dim) == EOF){
    printf("ERROR: '%s' potentially corrupted.\n", filename);
    exit(1);
  }

  size_t arr[num_layers+2];
  Nonlinearity logistics[num_layers+1];
  arr[0] = input_dim;
  for(int i = 1; i <= num_layers; i++){
    if(fscanf(fp, " %lu %u ", &arr[i], &logistics[i-1]) == EOF){
      printf("ERROR: '%s' potentially corrupted.\n", filename);
      exit(1);
    }
  }
  if(fscanf(fp, " %lu %u", &arr[num_layers+1], &logistics[num_layers]) == EOF){
    printf("ERROR: '%s' potentially corrupted.\n", filename);
    exit(1);
  }
  RNN n = rnn_from_arr(arr, num_layers+2);
  for(int i = 0; i < n.depth; i++){
    n.layers[i].logistic = logistics[i];
  }
  n.output_layer.logistic = logistics[num_layers];
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
