/* Author: Jonah Siekmann
 * Written 7/24/2018, updated 1/17/2019
 * This is a multilayer perceptron implementation. I've tested it with mnist and a few trivial problems.
 * Every function beginning with static is meant for internal use only. You may call any other function.
 */

#include <mlp.h>
#include <math.h>
#include <string.h>

#ifdef GPU
#include <opencl_utils.h>
#endif

#define ALLOCATE(TYPE, NUM) (TYPE*)malloc((NUM) * (sizeof(TYPE)));
#define PRINTLIST(name, len) printf("printing %s: [", #name); for(int xyz = 0; xyz < len; xyz++){printf("%5.4f", name[xyz]); if(xyz < len-1) printf(", "); else printf("]\n");}

#define MAX_GRAD 5
#define DEBUG 0

/*
 * Calculates the inner product of two vectors.
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
 */
void zero_init(float *x, size_t dim){
	for(int i = 0; i < dim; i++)
		x[i] = 0.0;
}


/*
 * Calculates the gradients wrt cost function given a label vector y.
 */
float quadratic_cost(float *o, const float *y, float *dest, size_t dim){
	float sum = 0;
	for(int i = 0; i < dim; i++){
		dest[i] = (y[i] - o[i]);
		sum += 0.5*(y[i]-o[i]) * (y[i]-o[i]);
	}
	return sum;
}

float cross_entropy_cost(float *o, const float *y, float *dest, size_t dim){
	float sum = 0;
	for(int i = 0; i < dim; i++){
		if(o[i] > 0.9999) o[i] = 0.9999;
		if(o[i] < 0.0001) o[i] = 0.0001;
		float grad = (y[i]/o[i]) - ((1-y[i])/(1-o[i]));
		if(isnan(grad)){
			printf("ERROR: cross_entropy_cost(): got a nan from y: %f, o: %f\n", y[i], o[i]);
			exit(1);
		}
#if DEBUG
		if(grad > MAX_GRAD){
			printf("WARNING: cross_entropy_cost(): cost gradient massive (%5.3f). Is there an issue with the label (%5.3f)?\n", grad, y[i]);
			grad = MAX_GRAD;
		}
		if(grad < -MAX_GRAD){
			printf("WARNING: cross_entropy_cost(): cost gradient massive (%5.3f). Is there an issue with the label (%5.3f)?\n", grad, y[i]);
			grad = -MAX_GRAD;
		}
#endif
		dest[i] = grad;
		sum += -(y[i] * log(o[i]) + (1-y[i]) * log(1-o[i]));
	}
	return sum;
}


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


float mlp_cost(MLP *n, float *y){
	return n->cost_fn(n->output, y, n->cost_gradient, n->output_dimension);
}

/********* BEGIN CPU-ONLY FUNCTIONS **********/
#ifndef GPU

/* 
 * Creates mlp layer for cpu
 */
MLP_layer cpu_create_MLP_layer(size_t input_dimension, size_t num_neurons, float *params, float *param_grad, Nonlinearity logistic){
	MLP_layer layer;

	Neuron* neurons = ALLOCATE(Neuron, num_neurons);

	int param_bound = num_neurons * input_dimension; //The number of parameters to read from network's param array
	int param_idx = 0;
	for(int i = 0; i < num_neurons; i++){
		neurons[i].bias = &params[param_idx];
		neurons[i].weights = &params[param_idx+1];

		//Xavier (or Xavier-like) bias+weight initialization
		xavier_init(&params[param_idx], input_dimension+1, num_neurons);

		neurons[i].bias_grad = &param_grad[param_idx];
		neurons[i].weight_grad = &param_grad[param_idx+1];
		param_idx += input_dimension + 1;
	}

	layer.z = ALLOCATE(float, num_neurons);
	layer.output = ALLOCATE(float, num_neurons);
	layer.gradient = ALLOCATE(float, input_dimension);

	layer.neurons = neurons;
	layer.size = num_neurons;
	layer.input_dimension = input_dimension;
	
	layer.logistic = logistic; //Set layer activation function
	return layer;
}

/*
 * A function called through the createMLP() macro that allows creation of a network with any arbitrary number of layers.
 */
MLP cpu_mlp_from_arr(size_t arr[], size_t size, int initialize){
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

	n.cost_gradient = (float*)malloc(n.output_dimension * sizeof(float));
	n.layers = ALLOCATE(MLP_layer, (size-1));
	n.cost_fn = cross_entropy_cost;

	int param_idx = 0;
	for(int i = 1; i < size; i++){

			MLP_layer l;
			size_t layer_size = arr[i];
			size_t input_dimension = arr[i-1];

			float *param_addr = &n.params[param_idx];
			float *grad_addr = &n.param_grad[param_idx];

			param_idx += layer_size * (input_dimension+1);

			if(i < size-1)
				l = cpu_create_MLP_layer(input_dimension, layer_size, param_addr, grad_addr, sigmoid);
			else
				l = cpu_create_MLP_layer(input_dimension, layer_size, param_addr, grad_addr, softmax);
			
			n.layers[i-1] = l;
	}
	n.output = n.layers[n.depth-1].output;
	return n;
}

/*
 * Does a forward pass for a single layer.
 */
void cpu_mlp_layer_forward(MLP_layer *l, float *x){
	l->input = x; //need to save pointer for backward pass
	for(int i = 0; i < l->size; i++){
		float *w = l->neurons[i].weights; 
		l->z[i] = inner_product(x, w, l->input_dimension) + *l->neurons[i].bias;
		if(l->logistic != softmax)
			l->output[i] = activate(l->z[i], l->logistic);
	}
	if(l->logistic == softmax){
		double sum = 0; //must be double
		for(int i = 0; i < l->size; i++)
			sum += exp(l->z[i]);

		for(int i = 0; i < l->size; i++)
			l->output[i] = exp(l->z[i]) / sum;
	}
}

/*
 * Does a forward pass for the entire network.
 */
void cpu_mlp_forward(MLP *n, float *input){
	float *x = input;
	for(int i = 0; i < n->depth; i++){
		MLP_layer *l = &n->layers[i];
		cpu_mlp_layer_forward(l, x); //Do forward pass for this layer
		x = l->output; //Use this layer's output as the next layer's input
	}
	n->guess = 0;
	for(int i = 0; i < n->output_dimension; i++)
		if(n->output[n->guess] < n->output[i])
			n->guess = i;
}

/*
 * Calculates the backward pass for a single layer (does parameter update)
 */
void cpu_mlp_layer_backward(MLP_layer *l, float *grads){
	float *avg_outs = l->output;
	for(int j = 0; j < l->input_dimension; j++)
		l->gradient[j] = 0;

	for(int i = 0; i < l->size; i++){
		float gradient = grads[i]; //gradient of this neuron's output with respect to cost
		float d_output = differentiate(avg_outs[i], l->logistic);

		for(int j = 0; j < l->input_dimension; j++){
			float w = l->neurons[i].weights[j];
			float d = d_output;
			float g = gradient;
			float x = l->input[j];

			l->gradient[j] += w * d * g;
			l->neurons[i].weight_grad[j] += x * d * g;
		}
		*l->neurons[i].bias_grad += gradient * d_output;
	}
}

/*
 * Does backward pass for entire network (calculates n.param_grad)
 */
void cpu_mlp_backward(MLP *n){

	float *grads = n->cost_gradient;
	for(int i = n->depth-1; i >= 0; i--){
		cpu_mlp_layer_backward(&n->layers[i], grads);
		grads = n->layers[i].gradient;
	}
	//printf("printing p %p\n", n->param_grad);
	//PRINTLIST(n->param_grad, 10);
	//getchar();
}

/*
 * Deallocates a network's memory from the heap
 */
void dealloc_mlp(MLP *n){
	int counter = 0;
	for(int i = 0; i < n->depth; i++){
		MLP_layer *l = &n->layers[i];
		free(l->output);
		free(l->gradient);
		free(l->neurons);
	}
	free(n->params);
	free(n->cost_gradient);
	free(n->layers);
}
#endif
/********* END CPU-ONLY FUNCTIONS **********/

/********* BEGIN GPU-ONLY FUNCTIONS **********/
#ifdef GPU

static cl_kernel mlp_forward_kernel;
static cl_kernel mlp_input_gradient_kernel, mlp_parameter_gradient_kernel;
static cl_kernel zero_init_kernel;

/* 
 * Creates mlp layer for gpu
 */
static int ARE_KERNELS_INITIALIZED = 0;
void mlp_kernel_setup(){
	char *kernels[] = {"include/nonlinear.h", "src/mlp.cl", "src/logistic.cl"};

	int err = 0;

	char *src = get_kernel_source(kernels, 3);
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

	ARE_KERNELS_INITIALIZED = 1;
}

MLP gpu_mlp_from_arr(size_t arr[], size_t size, int initialize){
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
	n.params = ALLOCATE(float, n.num_params);
	n.layers = ALLOCATE(MLP_layer, (size-1));

	int param_idx = 0;
	for(int i = 1; i < size; i++){
		MLP_layer l;
		l.size = arr[i];
		l.input_dimension = arr[i-1];
		l.gradient = clCreateBuffer(get_opencl_context(), CL_MEM_READ_WRITE, sizeof(float) * l.input_dimension, NULL, &err);
		check_error(err, "creating gradient buffer.");
		
		l.z = clCreateBuffer(get_opencl_context(), CL_MEM_READ_WRITE, sizeof(float) * l.size, NULL, &err);
		check_error(err, "creating linear buffer.");

		l.output = clCreateBuffer(get_opencl_context(), CL_MEM_READ_WRITE, sizeof(float) * l.size, NULL, &err);
		check_error(err, "creating output buffer.");

		if(i < size-1)
			l.logistic = sigmoid;
		else
			l.logistic = softmax;
		l.param_offset = param_idx;
		if(initialize){
			for(int j = 0; j < l.size; j++){
				xavier_init(&n.params[param_idx + j * (arr[i-1]+1)], arr[i-1]+1, arr[i]);
			}
		}
	
		n.layers[i-1] = l;
		param_idx += (arr[i-1]+1)*arr[i];
	}

	if(initialize){ //set initialize to zero if want to load in non-random params
		n.gpu_params = clCreateBuffer(get_opencl_context(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * n.num_params, n.params, &err);
		check_error(err, "creating & copying gpu params");
	}

	n.param_grad = clCreateBuffer(get_opencl_context(), CL_MEM_READ_WRITE, sizeof(float) * n.num_params, NULL, &err);
	check_error(err, "creating gpu param grads");

	//check_error(clSetKernelArg(zero_init_kernel, 0, sizeof(cl_mem), n.param_grad), "couldn't set zero init kernel arg 0");
	//check_error(clEnqueueNDRangeKernel(n.queue, zero_init_kernel, 1, NULL, &n.num_params, NULL, 0, NULL, NULL), "couldn't enqueue zero init kernel");


	n.network_input = clCreateBuffer(get_opencl_context(), CL_MEM_READ_WRITE, sizeof(float) * n.input_dimension, NULL, &err);
	check_error(err, "creating temp input buffer");

	n.network_grad = clCreateBuffer(get_opencl_context(), CL_MEM_READ_WRITE, sizeof(float) * n.output_dimension, NULL, &err);
	check_error(err, "creating network grad buffer on line ");

	n.output = ALLOCATE(float, n.output_dimension);
	n.cost_gradient = ALLOCATE(float, n.output_dimension);
	n.cost_fn = cross_entropy_cost;

	return n;
}

#define ARR_FROM_GPU(clmem, size, name) float name[size]; clEnqueueReadBuffer(get_opencl_context(), clmem, 1, 0, sizeof(float) * size, name, 0, NULL, NULL); 

void gpu_mlp_layer_forward(MLP_layer *l, cl_mem input, cl_mem params){
	l->input = input;
	int params_per_neuron = l->input_dimension + 1;
	check_error(clSetKernelArg(mlp_forward_kernel, 0, sizeof(cl_mem), &input), "setting forward kernel arg0");
	check_error(clSetKernelArg(mlp_forward_kernel, 1, sizeof(cl_mem), &l->z), "setting forward kernel arg1");
	check_error(clSetKernelArg(mlp_forward_kernel, 2, sizeof(cl_mem), &params), "setting forward kernel arg2");
	check_error(clSetKernelArg(mlp_forward_kernel, 3, sizeof(int), &l->input_dimension), "setting forward kernel arg3");
	check_error(clSetKernelArg(mlp_forward_kernel, 4, sizeof(int), &l->param_offset), "setting forward kernel arg4");
	check_error(clSetKernelArg(mlp_forward_kernel, 5, sizeof(int), &params_per_neuron), "setting forward kernel arg5");
	check_error(clEnqueueNDRangeKernel(get_opencl_queue(), mlp_forward_kernel, 1, NULL, &l->size, NULL, 0, NULL, NULL), "couldn't enqueue linear kernel");
	
	if(l->logistic != softmax){
		check_error(clSetKernelArg(logistic_kernel, 0, sizeof(cl_mem), &l->z), "setting logistic arg 0");
		check_error(clSetKernelArg(logistic_kernel, 1, sizeof(cl_mem), &l->output), "setting logistic arg 1");
		check_error(clSetKernelArg(logistic_kernel, 2, sizeof(Nonlinearity), &l->logistic), "setting logistic arg 2");
		check_error(clEnqueueNDRangeKernel(get_opencl_queue(), logistic_kernel, 1, NULL, &l->size, NULL, 0, NULL, NULL), "couldn't enqueue logistic kernel");
	}else{
		size_t one = 1;
		cl_mem smsum = get_softmax_sum(); //retrieve global softmax sum placeholder
		check_error(clSetKernelArg(softmax_sum_kernel, 0, sizeof(cl_mem), &l->z), "setting softmax sum arg 0");
		check_error(clSetKernelArg(softmax_sum_kernel, 1, sizeof(cl_mem), &smsum), "setting softmax sum arg 1");
		check_error(clSetKernelArg(softmax_sum_kernel, 2, sizeof(int), &l->size), "setting softmax sum arg 2");
		check_error(clEnqueueNDRangeKernel(get_opencl_queue(), softmax_sum_kernel, 1, NULL, &one, NULL, 0, NULL, NULL), "couldn't do softmax sum");

		check_error(clSetKernelArg(softmax_kernel, 0, sizeof(cl_mem), &l->z), "setting softmax arg 0");
		check_error(clSetKernelArg(softmax_kernel, 1, sizeof(cl_mem), &l->output), "setting softmax arg 1");
		check_error(clSetKernelArg(softmax_kernel, 2, sizeof(cl_mem), &smsum), "setting softmax arg 2");
		check_error(clEnqueueNDRangeKernel(get_opencl_queue(), softmax_kernel, 1, NULL, &l->size, NULL, 0, NULL, NULL), "couldn't do softmax sum");
	}


}

void gpu_mlp_forward(MLP *n, float *x){
	check_error(clEnqueueWriteBuffer(get_opencl_queue(), n->network_input, 0, 0, sizeof(float) * n->input_dimension, x, 0, NULL, NULL), "enqueuing network input");

	cl_mem input = n->network_input;
	for(int i = 0; i < n->depth; i++){
		MLP_layer *l = &n->layers[i];
		l->input = input;
		gpu_mlp_layer_forward(l, input, n->gpu_params);
		input = l->output;
	}
	
	clEnqueueReadBuffer(get_opencl_queue(), input, 1, 0, sizeof(float) * n->output_dimension, n->output, 0, NULL, NULL);
	n->guess = 0;
	for(int i = 0; i < n->output_dimension; i++)
		if(n->output[n->guess] < n->output[i])
			n->guess = i;
}

void gpu_mlp_layer_backward(MLP_layer *l, cl_mem grad, cl_mem params, cl_mem param_grad){
	check_error(clSetKernelArg(mlp_input_gradient_kernel, 0, sizeof(cl_mem), &grad), "setting input grad kernel arg 0");
	check_error(clSetKernelArg(mlp_input_gradient_kernel, 1, sizeof(cl_mem), &l->output), "setting input grad kernel arg 1");
	check_error(clSetKernelArg(mlp_input_gradient_kernel, 2, sizeof(cl_mem), &params), "setting input grad kernel arg 2");
	check_error(clSetKernelArg(mlp_input_gradient_kernel, 3, sizeof(cl_mem), &l->gradient), "setting input grad kernel arg 3");
	check_error(clSetKernelArg(mlp_input_gradient_kernel, 4, sizeof(Nonlinearity), &l->logistic), "setting input grad kernel arg 3");
	check_error(clSetKernelArg(mlp_input_gradient_kernel, 5, sizeof(int), &l->param_offset), "setting input grad kernel arg 4");
	check_error(clSetKernelArg(mlp_input_gradient_kernel, 6, sizeof(int), &l->size), "setting input grad kernel arg 5");
	check_error(clSetKernelArg(mlp_input_gradient_kernel, 7, sizeof(int), &l->input_dimension), "setting input grad kernel arg 6");
	check_error(clEnqueueNDRangeKernel(get_opencl_queue(), mlp_input_gradient_kernel, 1, NULL, &l->input_dimension, NULL, 0, NULL, NULL), "couldn't enqueue input grad kernel");

	check_error(clSetKernelArg(mlp_parameter_gradient_kernel, 0, sizeof(cl_mem), &grad), "setting param grad kernel arg 0");
	check_error(clSetKernelArg(mlp_parameter_gradient_kernel, 1, sizeof(cl_mem), &l->output), "setting param grad kernel arg 1");
	check_error(clSetKernelArg(mlp_parameter_gradient_kernel, 2, sizeof(cl_mem), &l->input), "setting param grad kernel arg 2");
	check_error(clSetKernelArg(mlp_parameter_gradient_kernel, 3, sizeof(cl_mem), &param_grad), "setting param grad kernel arg 3");
	check_error(clSetKernelArg(mlp_parameter_gradient_kernel, 4, sizeof(Nonlinearity), &l->logistic), "setting param grad kernel arg 4");
	check_error(clSetKernelArg(mlp_parameter_gradient_kernel, 5, sizeof(int), &l->param_offset), "setting param grad kernel arg 5");
	check_error(clSetKernelArg(mlp_parameter_gradient_kernel, 6, sizeof(int), &l->size), "setting param grad kernel arg 6");
	check_error(clSetKernelArg(mlp_parameter_gradient_kernel, 7, sizeof(int), &l->input_dimension), "setting param grad kernel arg 7");
	check_error(clEnqueueNDRangeKernel(get_opencl_queue(), mlp_parameter_gradient_kernel, 1, NULL, &l->size, NULL, 0, NULL, NULL), "couldn't enqueue param grad kernel");
}

void gpu_mlp_backward(MLP *n){
	check_error(clEnqueueWriteBuffer(get_opencl_queue(), n->network_grad, 0, 0, sizeof(float) * n->output_dimension, n->cost_gradient, 0, NULL, NULL), "enqueuing network grads");

	cl_mem grads = n->network_grad;
	int l_idx = n->depth;
	while(l_idx --> 0){ //l_idx goes to 0!
		MLP_layer *l = &n->layers[l_idx];
		gpu_mlp_layer_backward(l, grads, n->gpu_params, n->param_grad);
		grads = l->gradient;
	}
}

#endif

MLP mlp_from_arr(size_t arr[], size_t size, int initialize){
#ifndef GPU
	return cpu_mlp_from_arr(arr, size, initialize);
#else
	return gpu_mlp_from_arr(arr, size, initialize);
#endif
}

void mlp_forward(MLP *n, float *x){
#ifndef GPU
	cpu_mlp_forward(n, x);
#else
	gpu_mlp_forward(n, x);
#endif
}

void mlp_backward(MLP *n){
#ifndef GPU
	cpu_mlp_backward(n);
#else
	gpu_mlp_backward(n);
#endif
}

 /*
	* IO FUNCTIONS FOR READING AND WRITING TO A FILE
	*/

static void getWord(FILE *fp, char* dest){
	memset(dest, '\0', strlen(dest));
	//printf("bytes read: %lu\n", fread(dest, 1024, 1, fp));
	int res = fscanf(fp, " %1023s", dest);
}
/* 
 * Saves the network's state to a file that can be read later.
 * n: A pointer to the network.
 * filename: The desired filename and path.
 */
void save_mlp(MLP *n, const char* filename){
#ifdef GPU
	float *tmp = (float*)malloc(sizeof(float) * n->num_params);
	clEnqueueReadBuffer(get_opencl_queue(), n->gpu_params, 1, 0, sizeof(float) * n->num_params, tmp, 0, NULL, NULL);
#endif
	char buff[1024];
	memset(buff, '\0', 1024);

	//Create file
	FILE *fp = fopen(filename, "w");
	printf("Saving mlp to: %s\n", filename);
	memset(buff, '\0', strlen(buff));

	//Write header info to file
	fprintf(fp, "MLP %lu %lu ", n->depth, n->input_dimension);
	for(int i = 0; i < n->depth; i++){
		fprintf(fp, "%lu", n->layers[i].size);
		if(i < n->depth-1) fprintf(fp, " ");
		else fprintf(fp, "\n");
	}
	
	for(int i = 0; i < n->num_params; i++){
#ifdef GPU
		fprintf(fp, "%f", tmp[i]);
#else
		fprintf(fp, "%f", n->params[i]);
#endif
		if(i < n->num_params-1) fprintf(fp, " ");
		else fprintf(fp, "\n");
	}
	fclose(fp);
#ifdef GPU
	free(tmp);
#endif
}

/*
 * Loads a network from a file.
 * filename: The path to the file.
 */
MLP load_mlp(const char *filename){
	int f;
	FILE *fp = fopen(filename, "rb");
	char buff[1024];
	memset(buff, '\0', 1024);

	getWord(fp, buff); //Get first word to check if MLP file

	if(strcmp(buff, "MLP") != 0){
		printf("ERROR: [%s] is not MLP.\n", buff);
		exit(1);
	}
	size_t num_layers, input_dim;

	f = fscanf(fp, "%lu %lu", &num_layers, &input_dim);
	size_t arr[num_layers+1];
	arr[0] = input_dim;
	for(int i = 1; i <= num_layers; i++){
		f = fscanf(fp, " %lu", &arr[i]);
	}

	MLP n;
	n = mlp_from_arr(arr, num_layers+1, 0);
	for(int i = 0; i < n.num_params; i++){
		f = fscanf(fp, "%f", &n.params[i]);
	}
#ifdef GPU
	int err;
	n.gpu_params = clCreateBuffer(get_opencl_context(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * n.num_params, n.params, &err);
	check_error(err, "could not write params into gpu");
#endif
	return n;
}
