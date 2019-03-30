/* Author: Jonah Siekmann
 * Written 7/24/2018, updated 1/17/2019
 * This is a multilayer perceptron implementation. I've tested it with mnist and a few trivial problems.
 * Every function beginning with static is meant for internal use only. You may call any other function.
 */

#include <math.h>
#include <string.h>

#include <logistic.kernel>
#include <mlp.h>

#ifdef SIEKNET_USE_GPU
#include <opencl_utils.h>
#endif

#define ALLOCATE(TYPE, NUM) (TYPE*)malloc((NUM) * (sizeof(TYPE)));
#define PRINTLIST(name, len) printf("printing %s: [", #name); for(int xyz = 0; xyz < len; xyz++){printf("%5.4f", name[xyz]); if(xyz < len-1) printf(", "); else printf("]\n");}
#define ARR_FROM_SIEKNET_USE_GPU(name, gpumem, size) float name[size]; memset(name, '\0', size*sizeof(float)); check_error(clEnqueueReadBuffer(get_opencl_queue0(), gpumem, 1, 0, sizeof(float) * size, name, 0, NULL, NULL), "error reading from gpu (ARR_FROM_SIEKNET_USE_GPU)");

#ifdef SIEKNET_USE_GPU
static cl_kernel mlp_forward_kernel;
static cl_kernel mlp_input_gradient_kernel, mlp_parameter_gradient_kernel;
static cl_kernel softmax_sum_kernel, softmax_kernel;
static cl_kernel cost_kernel, cost_gradient_kernel;
static cl_kernel logistic_kernel, zero_init_kernel;

static int ARE_KERNELS_INITIALIZED = 0;
void mlp_kernel_setup(){
	char *kernels[] = {"include/logistic.h", "include/mlp.kernel", "include/logistic.kernel", "src/mlp.cl", "src/logistic.cl"};

	int err = 0;

	char *src = get_kernel_source(kernels, 5);
	cl_program prog = build_program(src);
	free(src);

	mlp_forward_kernel = clCreateKernel(prog, "mlp_forward_kernel", &err);
	check_error(err, "couldn't make forwards kernel");

	mlp_input_gradient_kernel = clCreateKernel(prog, "mlp_input_gradient_kernel", &err);
	check_error(err, "couldn't make mlp input grad kernel");
	
	mlp_parameter_gradient_kernel = clCreateKernel(prog, "mlp_parameter_gradient_kernel", &err);
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

/*
 * Calculates the inner product of two vectors.
 * DEPRECATED 
 */
float inner_product(const float *x, const float *y, size_t length){
	float sum = 0;
	for(long i = 0; i < length; i++){
		sum += x[i] * y[i];	
	}
	return sum;
}

/*
 * Does Xavier (or Xavier-like) parameter initialization
 */
void xavier_init(float *params, size_t input_dim, size_t layer_size){
	for(int i = 0; i < input_dim; i++){
		float rand_param = (((float)rand())/((float)RAND_MAX)) * sqrt(2.0 / (input_dim + layer_size));
		if(rand()&1) rand_param *= -1;
		params[i] = rand_param;
	}
}

/*
 * Does zero-init on a vector
 * DEPRECATED
 */
void zero_init(float *x, size_t dim){
	for(int i = 0; i < dim; i++)
		x[i] = 0.0;
}
#ifndef SIEKNET_USE_GPU
float cpu_cost(float *o, float *y, float *dest, size_t dim, Costfn c){
	float sum;
	agnostic_cost_kernel(o, y, &sum, dim, c);
	for(int i = 0; i < dim; i++){
		agnostic_cost_gradient_kernel(o, y, dest, c, i);
	}
	//PRINTLIST(dest, dim);
	return sum;
}
#else
float gpu_cost(cl_mem o, cl_mem y, cl_mem dest, size_t dim, Costfn c){
	const size_t one = 1;
	const int neurons = (int)dim;
	//cl_mem sum = get_cost_scalar();
	cl_mem sum = get_softmax_sum(); //retrieve global softmax sum placeholder
	check_error(clSetKernelArg(cost_kernel, 0, sizeof(cl_mem), &o), "setting CEC kernel arg 0");
	check_error(clSetKernelArg(cost_kernel, 1, sizeof(cl_mem), &y), "setting CEC kernel arg 1");
	check_error(clSetKernelArg(cost_kernel, 2, sizeof(cl_mem), &sum), "setting CEC kernel arg 2");
	check_error(clSetKernelArg(cost_kernel, 3, sizeof(int), &neurons), "setting CEC kernel arg 3");
	check_error(clSetKernelArg(cost_kernel, 4, sizeof(Costfn), &c), "setting CEC kernel arg 4");
	check_error(clEnqueueNDRangeKernel(get_opencl_queue0(), cost_kernel, 1, NULL, &one, NULL, 0, NULL, NULL), "couldn't enqueue cost scalar kernel");
	//check_error(clEnqueueNDRangeKernel(get_opencl_queue0(), softmax_sum_kernel, 1, NULL, &one, NULL, 0, NULL, NULL), "couldn't do softmax sum");

	check_error(clSetKernelArg(cost_gradient_kernel, 0, sizeof(cl_mem), &o), "setting CEC kernel arg 0");
	check_error(clSetKernelArg(cost_gradient_kernel, 1, sizeof(cl_mem), &y), "setting CEC kernel arg 1");
	check_error(clSetKernelArg(cost_gradient_kernel, 2, sizeof(cl_mem), &dest), "setting CEC kernel arg 2");
	check_error(clSetKernelArg(cost_gradient_kernel, 3, sizeof(Costfn), &c), "setting cost grad kernel arg 3");
	check_error(clEnqueueNDRangeKernel(get_opencl_queue0(), cost_gradient_kernel, 1, NULL, &dim, NULL, 0, NULL, NULL), "couldn't enqueue cost scalar kernel");

	float ret;
	check_error(clEnqueueReadBuffer(get_opencl_queue0(), sum, 1, 0, sizeof(float), &ret, 0, NULL, NULL), "error reading cost from gpu");
	check_error(clFinish(get_opencl_queue0()), "waiting for cost kernels to finish\n");
	return ret;

}
#endif

#ifndef SIEKNET_USE_GPU
float cpu_mlp_cost(MLP *n, float *y){
	float *o = n->layers[n->depth-1].output;
	float *dest = n->cost_gradient;
	return cpu_cost(o, y, dest, n->output_dimension, n->cost_fn);
}
#else
float gpu_mlp_cost(MLP *n, float *y){
	check_error(clEnqueueWriteBuffer(get_opencl_queue0(), n->output_label, 1, 0, sizeof(float) * n->output_dimension, y, 0, NULL, NULL), "enqueuing label");
	cl_mem o = n->layers[n->depth-1].output;
	cl_mem dest = n->cost_gradient;
	return gpu_cost(o, n->output_label, dest, n->output_dimension, n->cost_fn);
}
#endif

float mlp_cost(MLP *n, float *y){
#ifndef SIEKNET_USE_GPU
	return cpu_mlp_cost(n, y);
#else
	return gpu_mlp_cost(n, y);
#endif
}

/* 
 * Creates mlp layer for cpu
 */
#ifndef SIEKNET_USE_GPU
MLP_layer cpu_create_MLP_layer(const size_t input_dimension, const size_t num_neurons, float *params, const int param_idx, const Nonlinearity logistic){

	MLP_layer layer;

	layer.param_offset = param_idx;

	for(int i = 0; i < num_neurons; i++){
		//neurons[i].bias = &params[param_idx];
		//neurons[i].weights = &params[param_idx+1];

		//Xavier (or Xavier-like) bias+weight initialization
		xavier_init(&params[param_idx + i*(input_dimension+1)], input_dimension+1, num_neurons);

		//neurons[i].bias_grad = &param_grad[param_idx];
		//neurons[i].weight_grad = &param_grad[param_idx+1];
	}

	layer.z = ALLOCATE(float, num_neurons);
	layer.output = ALLOCATE(float, num_neurons);
	layer.input_gradient = ALLOCATE(float, input_dimension);

	//layer.neurons = neurons;
	layer.size = num_neurons;
	layer.input_dimension = input_dimension;
	
	layer.logistic = logistic; //Set layer activation function
	return layer;
}
#else
MLP_layer gpu_create_MLP_layer(size_t input_dim, size_t size, cl_mem params, int param_offset, Nonlinearity nonlin){
	MLP_layer l;
	l.input_dimension = input_dim;
	l.size = size;
	l.param_offset = param_offset;
	int err;
	l.input_gradient = clCreateBuffer(get_opencl_context(), CL_MEM_READ_WRITE, sizeof(float) * l.input_dimension, NULL, &err);
	check_error(err, "creating gradient buffer.");
	
	l.z = clCreateBuffer(get_opencl_context(), CL_MEM_READ_WRITE, sizeof(float) * l.size, NULL, &err);
	check_error(err, "creating linear buffer.");

	l.output = clCreateBuffer(get_opencl_context(), CL_MEM_READ_WRITE, sizeof(float) * l.size, NULL, &err);
	check_error(err, "creating output buffer.");

	l.logistic = nonlin;
	float *tmp = (float*)malloc(sizeof(float)*(input_dim+1)*size);
	for(int j = 0; j < l.size; j++){
		xavier_init(&tmp[j * (input_dim+1)], input_dim+1, l.size);
	}
	check_error(clEnqueueWriteBuffer(get_opencl_queue0(), params, 1, param_offset * sizeof(float), sizeof(float)*l.size*(l.input_dimension+1) , tmp, 0, NULL, NULL), "copying into gpu params");
	free(tmp);
	return l;
}
#endif

/*
 * A function called through the createMLP() macro that allows creation of a network with any arbitrary number of layers.
 */
#ifndef SIEKNET_USE_GPU
MLP cpu_mlp_from_arr(size_t arr[], size_t size){
	MLP n;
	n.input_dimension = arr[0];
	n.output_dimension = arr[size-1];
	n.depth = size-1;

	size_t num_params = 0;
	size_t num_outputs = 0;
	for(int i = 1; i < size; i++){
		num_params += (arr[i-1]+1)*arr[i];
		num_outputs += arr[i];
	}

	n.num_params = num_params;
	n.params = ALLOCATE(float, num_params);

	n.param_grad = ALLOCATE(float, num_params);

	//n.cost_gradient = (float*)malloc(n.output_dimension * sizeof(float));
	n.cost_gradient = ALLOCATE(float, n.output_dimension);
	n.layers = ALLOCATE(MLP_layer, (size-1));
	n.cost_fn = cross_entropy;

	int param_idx = 0;
	for(int i = 1; i < size; i++){
			MLP_layer l;
			if(i < size-1)
				l = cpu_create_MLP_layer(arr[i-1], arr[i], n.params, param_idx, sigmoid);
			else
				l = cpu_create_MLP_layer(arr[i-1], arr[i], n.params, param_idx, softmax);

			n.layers[i-1] = l;
			param_idx += arr[i] * (arr[i-1]+1);
	}
	n.output = n.layers[n.depth-1].output;
	return n;
}
#else
MLP gpu_mlp_from_arr(size_t arr[], size_t size){
	initialize_opencl();
	if(!ARE_KERNELS_INITIALIZED)
		mlp_kernel_setup();

	MLP n;
	n.input_dimension = arr[0];
	n.output_dimension = arr[size-1];
	n.depth = size-1;

	n.num_params = 0;

	for(int i = 1; i < size; i++){
		n.num_params += (arr[i-1]+1)*arr[i];
	}

	int err = 0;
	n.layers = ALLOCATE(MLP_layer, (size-1));

	n.params = clCreateBuffer(get_opencl_context(), CL_MEM_READ_WRITE, sizeof(float) * n.num_params, NULL, &err);
	check_error(err, "creating & copying gpu params");

	int param_idx = 0;
	for(int i = 1; i < size; i++){
		MLP_layer l;
		if(i < size-1)
			l = gpu_create_MLP_layer(arr[i-1], arr[i], n.params, param_idx, sigmoid);
		else
			l = gpu_create_MLP_layer(arr[i-1], arr[i], n.params, param_idx, softmax);

		n.layers[i-1] = l;
		param_idx += (arr[i-1]+1)*arr[i];
	}

	n.param_grad = clCreateBuffer(get_opencl_context(), CL_MEM_READ_WRITE, sizeof(float) * n.num_params, NULL, &err);
	check_error(err, "creating gpu param grads");

	check_error(clSetKernelArg(zero_init_kernel, 0, sizeof(cl_mem), &n.param_grad), "couldn't set zero init kernel arg 0");
	check_error(clEnqueueNDRangeKernel(get_opencl_queue0(), zero_init_kernel, 1, NULL, &n.num_params, NULL, 0, NULL, NULL), "couldn't enqueue zero init kernel");

	n.network_input = clCreateBuffer(get_opencl_context(), CL_MEM_READ_WRITE, sizeof(float) * n.input_dimension, NULL, &err);
	check_error(err, "creating temp input buffer");

	//n.cost_gradient = ALLOCATE(float, n.output_dimension);
	n.cost_gradient = clCreateBuffer(get_opencl_context(), CL_MEM_READ_WRITE, sizeof(float) * n.output_dimension, NULL, &err);
	check_error(err, "creating network grad buffer");

	n.output_label = clCreateBuffer(get_opencl_context(), CL_MEM_READ_WRITE, sizeof(float) * n.output_dimension, NULL, &err);
	check_error(err, "creating output label buffer");

	n.output = ALLOCATE(float, n.output_dimension);
	n.cost_fn = cross_entropy;

	return n;
}
#endif

/*
 * Does a forward pass for a single layer.
 */
#ifndef SIEKNET_USE_GPU
void cpu_mlp_layer_forward(MLP_layer *l, float *input, float *params){
	l->input = input; //need to save pointer for backward pass
	for(int i = 0; i < l->size; i++){
		agnostic_mlp_forward_kernel(input, l->z, params, l->input_dimension, l->param_offset, (l->input_dimension+1), i);
	}
	if(l->logistic != softmax){
		for(int i = 0; i < l->size; i++){
			agnostic_logistic_kernel(l->z, l->output, l->logistic, i);
		}
	}else{
		float smsum;
		agnostic_softmax_sum_kernel(l->z, &smsum, l->size);
		for(int i = 0; i < l->size; i++){
			agnostic_softmax_kernel(l->z, l->output, &smsum, i);
		}
	}
}
#else
void gpu_mlp_layer_forward(MLP_layer *l, cl_mem input, cl_mem params){
	l->input = input;
	int params_per_neuron = l->input_dimension + 1;
	check_error(clSetKernelArg(mlp_forward_kernel, 0, sizeof(cl_mem), &input), "setting forward kernel arg0");
	check_error(clSetKernelArg(mlp_forward_kernel, 1, sizeof(cl_mem), &l->z), "setting forward kernel arg1");
	check_error(clSetKernelArg(mlp_forward_kernel, 2, sizeof(cl_mem), &params), "setting forward kernel arg2");
	check_error(clSetKernelArg(mlp_forward_kernel, 3, sizeof(int), &l->input_dimension), "setting forward kernel arg3");
	check_error(clSetKernelArg(mlp_forward_kernel, 4, sizeof(int), &l->param_offset), "setting forward kernel arg4");
	check_error(clSetKernelArg(mlp_forward_kernel, 5, sizeof(int), &params_per_neuron), "setting forward kernel arg5");
	check_error(clEnqueueNDRangeKernel(get_opencl_queue0(), mlp_forward_kernel, 1, NULL, &l->size, NULL, 0, NULL, NULL), "gpu_mlp_layer_forward(): couldn't enqueue linear kernel");
	
	if(l->logistic != softmax){
		check_error(clSetKernelArg(logistic_kernel, 0, sizeof(cl_mem), &l->z), "setting logistic arg 0");
		check_error(clSetKernelArg(logistic_kernel, 1, sizeof(cl_mem), &l->output), "setting logistic arg 1");
		check_error(clSetKernelArg(logistic_kernel, 2, sizeof(Nonlinearity), &l->logistic), "setting logistic arg 2");
		check_error(clEnqueueNDRangeKernel(get_opencl_queue0(), logistic_kernel, 1, NULL, &l->size, NULL, 0, NULL, NULL), "gpu_mlp_layer_forward(): couldn't enqueue logistic kernel");
	}else{
		size_t one = 1;
		cl_mem smsum = get_softmax_sum(); //retrieve global softmax sum placeholder
		check_error(clSetKernelArg(softmax_sum_kernel, 0, sizeof(cl_mem), &l->z), "setting softmax sum arg 0");
		check_error(clSetKernelArg(softmax_sum_kernel, 1, sizeof(cl_mem), &smsum), "setting softmax sum arg 1");
		check_error(clSetKernelArg(softmax_sum_kernel, 2, sizeof(int), &l->size), "setting softmax sum arg 2");
		check_error(clEnqueueNDRangeKernel(get_opencl_queue0(), softmax_sum_kernel, 1, NULL, &one, NULL, 0, NULL, NULL), "couldn't do softmax sum");

		check_error(clSetKernelArg(softmax_kernel, 0, sizeof(cl_mem), &l->z), "setting softmax arg 0");
		check_error(clSetKernelArg(softmax_kernel, 1, sizeof(cl_mem), &l->output), "setting softmax arg 1");
		check_error(clSetKernelArg(softmax_kernel, 2, sizeof(cl_mem), &smsum), "setting softmax arg 2");
		check_error(clEnqueueNDRangeKernel(get_opencl_queue0(), softmax_kernel, 1, NULL, &l->size, NULL, 0, NULL, NULL), "couldn't do softmax sum");
	}
	check_error(clFinish(get_opencl_queue0()), "waiting for logistic kernels to finish");
}
#endif

/*
 * Does a forward pass for the entire network.
 */
#ifndef SIEKNET_USE_GPU
void cpu_mlp_forward(MLP *n, float *input){
	float *x = input;
	for(int i = 0; i < n->depth; i++){
		MLP_layer *l = &n->layers[i];
		cpu_mlp_layer_forward(l, x, n->params); //Do forward pass for this layer
		x = l->output; //Use this layer's output as the next layer's input
	}
	n->guess = 0;
	for(int i = 0; i < n->output_dimension; i++)
		if(n->output[n->guess] < n->output[i])
			n->guess = i;
}
#else
void gpu_mlp_forward(MLP *n, float *x){
	check_error(clEnqueueWriteBuffer(get_opencl_queue0(), n->network_input, 0, 0, sizeof(float) * n->input_dimension, x, 0, NULL, NULL), "enqueuing network input");

	cl_mem input = n->network_input;
	for(int i = 0; i < n->depth; i++){
		MLP_layer *l = &n->layers[i];
		l->input = input;
		gpu_mlp_layer_forward(l, input, n->params);
		input = l->output;
	}
	
	check_error(clEnqueueReadBuffer(get_opencl_queue0(), input, 1, 0, sizeof(float) * n->output_dimension, n->output, 0, NULL, NULL), "reading network output");
	check_error(clFinish(get_opencl_queue0()), "couldn't wait for queue0 to end");

	n->guess = 0;
	for(int i = 0; i < n->output_dimension; i++)
		if(n->output[n->guess] < n->output[i])
			n->guess = i;
}
#endif


/*
 * Calculates the backward pass for a single layer (does parameter update)
 */
#ifndef SIEKNET_USE_GPU
void cpu_mlp_layer_backward(MLP_layer *l, float *grads, float *params, float *param_grad){
	for(int i = 0; i < l->input_dimension; i++){
		agnostic_mlp_input_gradient_kernel(grads, l->output, params, l->input_gradient, l->logistic, l->param_offset, l->size, l->input_dimension, i);
	}
	for(int i = 0; i < l->size; i++){
		agnostic_mlp_parameter_gradient_kernel(grads, l->output, l->input, param_grad, l->logistic, l->param_offset, l->size, l->input_dimension, i);
	}
}
#else
void gpu_mlp_layer_backward(MLP_layer *l, cl_mem grad, cl_mem params, cl_mem param_grad){
	//printf("setting mlp backward kernel args\n");
	check_error(clSetKernelArg(mlp_input_gradient_kernel, 0, sizeof(cl_mem), &grad), "setting input grad kernel arg 0");
	check_error(clSetKernelArg(mlp_input_gradient_kernel, 1, sizeof(cl_mem), &l->output), "setting input grad kernel arg 1");
	check_error(clSetKernelArg(mlp_input_gradient_kernel, 2, sizeof(cl_mem), &params), "setting input grad kernel arg 2");
	check_error(clSetKernelArg(mlp_input_gradient_kernel, 3, sizeof(cl_mem), &l->input_gradient), "setting input grad kernel arg 3");
	check_error(clSetKernelArg(mlp_input_gradient_kernel, 4, sizeof(Nonlinearity), &l->logistic), "setting input grad kernel arg 3");
	check_error(clSetKernelArg(mlp_input_gradient_kernel, 5, sizeof(int), &l->param_offset), "setting input grad kernel arg 4");
	check_error(clSetKernelArg(mlp_input_gradient_kernel, 6, sizeof(int), &l->size), "setting input grad kernel arg 5");
	check_error(clSetKernelArg(mlp_input_gradient_kernel, 7, sizeof(int), &l->input_dimension), "setting input grad kernel arg 6");
	//printf("enqueueing input grad kernel\n");
	check_error(clEnqueueNDRangeKernel(get_opencl_queue0(), mlp_input_gradient_kernel, 1, NULL, &l->input_dimension, NULL, 0, NULL, NULL), "couldn't enqueue input grad kernel");
	
	//printf("setting mlp param grad kernel args\n");
	check_error(clSetKernelArg(mlp_parameter_gradient_kernel, 0, sizeof(cl_mem), &grad), "setting param grad kernel arg 0");
	check_error(clSetKernelArg(mlp_parameter_gradient_kernel, 1, sizeof(cl_mem), &l->output), "setting param grad kernel arg 1");
	check_error(clSetKernelArg(mlp_parameter_gradient_kernel, 2, sizeof(cl_mem), &l->input), "setting param grad kernel arg 2");
	check_error(clSetKernelArg(mlp_parameter_gradient_kernel, 3, sizeof(cl_mem), &param_grad), "setting param grad kernel arg 3");
	check_error(clSetKernelArg(mlp_parameter_gradient_kernel, 4, sizeof(Nonlinearity), &l->logistic), "setting param grad kernel arg 4");
	check_error(clSetKernelArg(mlp_parameter_gradient_kernel, 5, sizeof(int), &l->param_offset), "setting param grad kernel arg 5");
	check_error(clSetKernelArg(mlp_parameter_gradient_kernel, 6, sizeof(int), &l->size), "setting param grad kernel arg 6");
	check_error(clSetKernelArg(mlp_parameter_gradient_kernel, 7, sizeof(int), &l->input_dimension), "setting param grad kernel arg 7");
	//printf("enqueueing param grad kerenl\n");
	check_error(clEnqueueNDRangeKernel(get_opencl_queue0(), mlp_parameter_gradient_kernel, 1, NULL, &l->size, NULL, 0, NULL, NULL), "couldn't enqueue param grad kernel");
	
	//printf("calling clfinsihg\n");
	check_error(clFinish(get_opencl_queue0()), "waiting for queue 0 to finish executing (forward pass)");
	//printf("done!\n");
}
#endif

/*
 * Does backward pass for entire network (calculates n.param_grad)
 */
#ifndef SIEKNET_USE_GPU
void cpu_mlp_backward(MLP *n){

	float *grads = n->cost_gradient;
	for(int i = n->depth-1; i >= 0; i--){
		cpu_mlp_layer_backward(&n->layers[i], grads, n->params, n->param_grad);
		grads = n->layers[i].input_gradient;
	}
	//printf("printing p %p\n", n->param_grad);
	//PRINTLIST(n->param_grad, 10);
	//getchar();
}
#else
void gpu_mlp_backward(MLP *n){
	//check_error(clEnqueueWriteBuffer(get_opencl_queue0(), n->network_grad, 1, 0, sizeof(float) * n->output_dimension, n->cost_gradient, 0, NULL, NULL), "enqueuing network grads");

	cl_mem grads = n->cost_gradient;
	int l_idx = n->depth;
	while(l_idx --> 0){ //l_idx goes to 0
		MLP_layer *l = &n->layers[l_idx];
		gpu_mlp_layer_backward(l, grads, n->params, n->param_grad);
		grads = l->input_gradient;
	}
}
#endif

/*
 * Deallocates a network's memory from the heap
 */
void dealloc_mlp(MLP *n){
#ifndef SIEKNET_USE_GPU
	for(int i = 0; i < n->depth; i++){
		MLP_layer *l = &n->layers[i];
		free(l->input_gradient);
		free(l->output);
		free(l->z);
	}
	free(n->params);
	free(n->param_grad);
	free(n->cost_gradient);
#else
	for(int i = 0; i < n->depth; i++){
		MLP_layer *l = &n->layers[i];
		check_error(clReleaseMemObject(l->input_gradient), "freeing clmem");
		check_error(clReleaseMemObject(l->z), "freeing clmem");
		check_error(clReleaseMemObject(l->output), "freeing clmem");
	}
	check_error(clReleaseMemObject(n->params), "freeing clmem");
	check_error(clReleaseMemObject(n->param_grad), "freeing clmem");
	check_error(clReleaseMemObject(n->network_input), "freeing clmem");
	check_error(clReleaseMemObject(n->output_label), "freeing clmem");
	check_error(clReleaseMemObject(n->cost_gradient), "freeing clmem");

#endif
	free(n->layers);
}

MLP mlp_from_arr(size_t arr[], size_t size){
#ifndef SIEKNET_USE_GPU
	return cpu_mlp_from_arr(arr, size);
#else
	return gpu_mlp_from_arr(arr, size);
#endif
}

void mlp_forward(MLP *n, float *x){
#ifndef SIEKNET_USE_GPU
	cpu_mlp_forward(n, x);
#else
	gpu_mlp_forward(n, x);
#endif
}

void mlp_backward(MLP *n){
#ifndef SIEKNET_USE_GPU
	cpu_mlp_backward(n);
#else
	gpu_mlp_backward(n);
#endif
}

 /*
	* IO FUNCTIONS FOR READING AND WRITING TO A FILE
	*/

static int getWord(FILE *fp, char* dest){
	memset(dest, '\0', strlen(dest));
	//printf("bytes read: %lu\n", fread(dest, 1024, 1, fp));
	return fscanf(fp, " %1023s", dest);
}
/* 
 * Saves the network's state to a file that can be read later.
 * n: A pointer to the network.
 * filename: The desired filename and path.
 */
void save_mlp(MLP *n, const char* filename){
#ifdef SIEKNET_USE_GPU
	float *tmp = (float*)malloc(sizeof(float) * n->num_params);
	clEnqueueReadBuffer(get_opencl_queue0(), n->params, 1, 0, sizeof(float) * n->num_params, tmp, 0, NULL, NULL);
#endif
	char buff[1024];
	memset(buff, '\0', 1024);

	//Create file
	FILE *fp = fopen(filename, "w");
	memset(buff, '\0', strlen(buff));

	//Write header info to file
	fprintf(fp, "MLP %lu %lu ", n->depth, n->input_dimension);
	for(int i = 0; i < n->depth; i++){
		fprintf(fp, "%lu", n->layers[i].size);
		if(i < n->depth-1) fprintf(fp, " ");
		else fprintf(fp, "\n");
	}
	
	for(int i = 0; i < n->num_params; i++){
#ifdef SIEKNET_USE_GPU
		fprintf(fp, "%f", tmp[i]);
#else
		fprintf(fp, "%f", n->params[i]);
#endif
		if(i < n->num_params-1) fprintf(fp, " ");
		else fprintf(fp, "\n");
	}
	fclose(fp);
#ifdef SIEKNET_USE_GPU
	free(tmp);
#endif
}

/*
 * Loads a network from a file.
 * filename: The path to the file.
 */
MLP load_mlp(const char *filename){
	FILE *fp = fopen(filename, "rb");
	char buff[1024];
	memset(buff, '\0', 1024);

	getWord(fp, buff); //Get first word to check if MLP file

	if(strcmp(buff, "MLP") != 0){
		printf("ERROR: [%s] is not MLP.\n", buff);
		exit(1);
	}
	size_t num_layers, input_dim;

	if(fscanf(fp, "%lu %lu", &num_layers, &input_dim) == EOF){
		printf("ERROR: '%s' corrupted.\n", filename);
		exit(1);
	}
	size_t arr[num_layers+1];
	arr[0] = input_dim;
	for(int i = 1; i <= num_layers; i++){
		if(fscanf(fp, " %lu", &arr[i]) == EOF){
			printf("ERROR: '%s' corrupted.\n", filename);
			exit(1);
		}
	}

	MLP n;
	n = mlp_from_arr(arr, num_layers+1);
#ifndef SIEKNET_USE_GPU
	for(int i = 0; i < n.num_params; i++){
		if(fscanf(fp, "%f", &n.params[i]) == EOF){
			printf("ERROR: '%s' corrupted.\n", filename);
			exit(1);
		}
	}
#else
	float *tmp = (float*)malloc(sizeof(float)*n.num_params);
	for(int i = 0; i < n.num_params; i++){
		if(fscanf(fp, "%f", &tmp[i]) == EOF){
			printf("ERROR: '%s' corrupted.\n", filename);
			exit(1);
		}
	}
	int err = 0;
	check_error(clEnqueueWriteBuffer(get_opencl_queue0(), n.params, 1, 0, sizeof(float)*n.num_params, tmp, 0, NULL, NULL), "could not enqueue layer params");
	check_error(err, "could not write params into gpu");
	free(tmp);
		//printf("n.layers[2].input_dimension: %d\n", n.layers[2].input_dimension);
		//printf("tf: %d\n", assert_equal(tmp, g2, n.layers[2].input_dimension));
#endif
	return n;
}
