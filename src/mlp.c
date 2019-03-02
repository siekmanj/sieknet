/* Author: Jonah Siekmann
 * Written 7/24/2018, updated 1/17/2019
 * This is a multilayer perceptron implementation. I've tested it with mnist and a few trivial problems.
 * Every function beginning with static is meant for internal use only. You may call any other function.
 */

#include <mlp.h>
#include <math.h>
#include <string.h>
#include <nonlinear.h>

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
 * Calculates the activation of a given neuron using softmax.
 */
void softmax(const float *z, float *dest, size_t dim){
	double sum = 0;
	for(int i = 0; i < dim; i++)
		sum += exp(z[i]);

	for(int i = 0; i < dim; i++){
		dest[i] = exp(z[i]) / sum;
		if(isnan(dest[i])){
			printf("ERROR: softmax(): nan from exp(%6.5f) / %6.5f\n", z[i], sum);
			exit(1);
		}
	}
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
 * Calculates the activations of a layer with sigmoid.
 */
void sigmoid(const float *z, float *dest, size_t dim){
	for(int i = 0; i < dim; i++){
		dest[i] = SIGMOID(z[i]);
		if(isnan(dest[i])){
			printf("ERROR: sigmoid(): nan from 1 / (1 + exp(-%6.5f))\n", z[i]);
			exit(1);
		}
	}
}

/*
 * Calculates the activations of a layer using ReLu.
 */
void relu(const float *z, float *dest, size_t dim){
	for(int i = 0; i < dim; i++){
		float x = z[i];
		if(x < 0) dest[i] = 0;
		else dest[i] = x;
	}
}

/*
 * Calculates the activations of a layer using tanh.
 */
void hypertan(const float *z, float *dest, size_t dim){
	for(int i = 0; i < dim; i++){
		float x = z[i];
		if(x > 7.0) dest[i] = 0.999998;
		else if(x < -7.0) dest[i] = -0.999998;
		else dest[i] = ((exp(x) - exp(-x))/(exp(x) + exp(-x)));
	}
}

/*
 * Calculates logistic function derivatives in terms of logistic output
 */
float differentiate(const float x, void (*logistic)(const float *, float *, size_t)){
	if(logistic == hypertan)
		return 1 - x*x;
	if(logistic == softmax || logistic == sigmoid)
		return x * (1 - x);
	if(logistic == relu){
		if(x > 0) return 1;
		else return 0;
	}
		
	printf("ERROR: differentiate(): derivative of logistic function not implemented!\n");
	exit(1);
}

/* 
 * Creates mlp layer for cpu
 */
MLP_layer cpu_create_MLP_layer(size_t input_dimension, size_t num_neurons, float *params, float *param_grad, void(*logistic)(const float *, float *, size_t)){
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
	}
	l->logistic(l->z, l->output, l->size); //Apply this layer's logistic function
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
 * Propagates gradients throughout network using the chain rule (does not do parameter update)
 */
void cpu_propagate_gradients(MLP *n, float *gradient){
	float *grads = gradient;
	for(int i = n->depth-1; i >= 0; i--){
		MLP_layer *l = &n->layers[i];
		
		float *avg = l->output;

		for(int j = 0; j < l->input_dimension; j++){
			float sum = 0;
			for(int k = 0; k < l->size; k++){
				float w = l->neurons[k].weights[j];
				float d = differentiate(avg[k], l->logistic);
				float g = grads[k];
				sum += w * d * g;
			}
			l->gradient[j] = sum;
		}
		grads = l->gradient;
	}
}

/*
 * Calculates the backward pass for a single layer (does parameter update)
 */
void cpu_mlp_layer_backward(MLP_layer *l, float *grads){
	float *avg_outs = l->output;
	for(int i = 0; i < l->size; i++){
		float gradient = grads[i]; //gradient of this neuron's output with respect to cost
		float d_output = differentiate(avg_outs[i], l->logistic);

		for(int j = 0; j < l->input_dimension; j++){
			float x = l->input[j];
			l->neurons[i].weight_grad[j] += gradient * d_output * x;
		}
		*l->neurons[i].bias_grad += gradient * d_output;
	}
}

/*
 * Does backward pass for entire network (calculates n.param_grad)
 */
void cpu_mlp_backward(MLP *n){

	float *grads = n->cost_gradient;
	cpu_propagate_gradients(n, grads);
	for(int i = n->depth-1; i >= 0; i--){
		cpu_mlp_layer_backward(&n->layers[i], grads);
		grads = n->layers[i].gradient;
	}
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
void check_error(int err, char *str){
  if(err != CL_SUCCESS){
    printf("ERROR: '%s': ", str);
    switch(err){
      case CL_INVALID_PROGRAM:
        printf("CL_INVALID_PROGRAM.\n");
        break;
      case CL_INVALID_PROGRAM_EXECUTABLE:
        printf("CL_INVALID_PROGRAM_EXECUTABLE.\n");
        break;
      case CL_INVALID_KERNEL_NAME:
        printf("CL_INVALID_KERNEL_NAME.\n");
        break;
      case CL_INVALID_KERNEL_DEFINITION:
        printf("CL_INVALID_KERNEL_DEFINITION.\n");
        break;
      case CL_INVALID_VALUE:
        printf("CL_INVALID_VALUE.\n");
        break;
      case CL_OUT_OF_HOST_MEMORY:
        printf("CL_OUT_OF_HOST_MEMORY.\n");
        break;
      case CL_INVALID_COMMAND_QUEUE:
        printf("CL_INVALID_COMMAND_QUEUE.\n");
        break;
      case CL_INVALID_KERNEL:
        printf("CL_INVALID_KERNEL.\n");
        break;
			case CL_INVALID_ARG_INDEX:
				printf("CL_INVALID_ARG_INDEX.\n");
				break;
			case CL_INVALID_ARG_VALUE:
				printf("CL_INVALID_ARG_VALUE.\n");
				break;
			case CL_INVALID_MEM_OBJECT:
				printf("CL_INVALID_MEM_OBJECT.\n");
				break;
			case CL_INVALID_ARG_SIZE:
				printf("CL_INVALID_ARG_SIZE.\n");
				break;
      case CL_INVALID_CONTEXT:
        printf("CL_INVALID_CONTEXT.\n");
        break;
      case CL_INVALID_KERNEL_ARGS:
        printf("CL_INVALID_KERNEL_ARGS.\n");
        break;
      case CL_INVALID_WORK_DIMENSION:
        printf("CL_INVALID_WORK_DIMENSION.\n");
        break;
      case CL_INVALID_WORK_GROUP_SIZE:
        printf("CL_INVALID_WORK_GROUP_SIZE.\n");
        break;
      case CL_INVALID_WORK_ITEM_SIZE:
        printf("CL_INVALID_WORK_ITEM_SIZE.\n");
        break;
      case CL_INVALID_GLOBAL_OFFSET:
        printf("CL_INVALID_GLOBAL_OFFSET.\n");
        break;
			case CL_INVALID_DEVICE:
				printf("CL_INVALID_DEVICE.\n");
				break;
			case CL_INVALID_BINARY:
				printf("CL_INVALID_BINARY.\n");
				break;
			case CL_INVALID_BUILD_OPTIONS:
				printf("CL_INVALID_BUILD_OPTIONS.\n");
				break;
			case CL_INVALID_OPERATION:
				printf("CL_INVALID_OPERATION.\n");
				break;
			case CL_COMPILER_NOT_AVAILABLE:
				printf("CL_COMPILER_NOT_AVAILABLE.\n");
				break;
			case CL_BUILD_PROGRAM_FAILURE:
				printf("CL_BUILD_PROGRAM_FAILURE.\n");
				break;
      default:
        printf("default err.\n");
        break;
    }
    exit(1);
  }
}


static cl_context SIEKNET_GLOBAL_CONTEXT;
static cl_command_queue SIEKNET_GLOBAL_QUEUE;

static cl_kernel linear;

/* 
 * Creates mlp layer for gpu
 */

static cl_context create_opencl_context(){
	cl_uint num_platforms, num_devices;
	int status = clGetPlatformIDs(0, NULL, &num_platforms);

	cl_platform_id platforms[num_platforms];

	check_error(clGetPlatformIDs(num_platforms, platforms, NULL), "couldn't get platforms");
	check_error(clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices), "couldn't count devices");
	
	cl_device_id devices[num_devices];

	check_error(clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, num_devices, devices, NULL), "couldn't get device ids.");

	const cl_context_properties cfg[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[0], 0, 0};

	cl_context context = clCreateContext(cfg, num_devices, devices, NULL, NULL, &status);

	check_error(status, "couldn't make context");
	printf("successfully created opencl context.\n");
	return context;
}

static cl_command_queue make_opencl_queue(cl_context c){
	cl_uint num_platforms, num_devices;
	int status = clGetPlatformIDs(0, NULL, &num_platforms);

	cl_platform_id platforms[num_platforms];

	check_error(clGetPlatformIDs(num_platforms, platforms, NULL), "couldn't get platform ids");
	check_error(clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices), "couldn't count gpu device ids");
	
	cl_device_id devices[num_devices];

	check_error(clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, num_devices, devices, NULL), "couldn't get gpu devices");

	int err;
	cl_command_queue queue = clCreateCommandQueue(c, devices[0], 0, &err);

	return queue;
}

void gpu_setup(){
	SIEKNET_GLOBAL_CONTEXT = create_opencl_context();
	SIEKNET_GLOBAL_QUEUE = make_opencl_queue(SIEKNET_GLOBAL_CONTEXT);

	char *forward_kernel = "../src/forward_kernel.cl";
	char *backward_kernel ="../src/backward_kernel.cl";

	FILE *fp = fopen(forward_kernel, "rb");
	if(!fp){
		printf("couldn't find '%s'.\n", forward_kernel);
		exit(1);
	}
	fseek(fp, 0, SEEK_END);
	size_t kernelfilelen = ftell(fp);
	fclose(fp);
	
	char *clfile = (char*)malloc(sizeof(char) * (kernelfilelen + 1));
	fp = fopen(forward_kernel, "rb");
	for(int i = 0; i < kernelfilelen; i++){
		clfile[i] = fgetc(fp);
	}
	fclose(fp);
	clfile[kernelfilelen] = '\0';

	int err = 0;
	cl_program forward = clCreateProgramWithSource(SIEKNET_GLOBAL_CONTEXT, 1, (const char**)&clfile, NULL, &err);
	check_error(err, "couldn't create program");

	check_error(clBuildProgram(forward, 0, NULL, NULL, NULL, NULL), "couldn't build!");

	linear = clCreateKernel(forward, "linear_kernel", &err);
	check_error(err, "couldn't make linear kern");

	sigmoid = clCreateKernel(forward, "sigmoid_kernel", &err);
	check_error(err, "couldn't make sigmoid kernel");

	printf("made it to the end of gpu setup!\n");

}

MLP gpu_mlp_from_arr(size_t arr[], size_t size){
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
		l.gradient = clCreateBuffer(SIEKNET_GLOBAL_CONTEXT, CL_MEM_READ_WRITE, sizeof(float) * l.input_dimension, NULL, &err);
		check_error(err, "creating gradient buffer.");
		
		l.z = clCreateBuffer(SIEKNET_GLOBAL_CONTEXT, CL_MEM_READ_WRITE, sizeof(float) * l.size, NULL, &err);
		check_error(err, "creating linear buffer.");

		l.output = clCreateBuffer(SIEKNET_GLOBAL_CONTEXT, CL_MEM_READ_WRITE, sizeof(float) * l.size, NULL, &err);
		check_error(err, "creating output buffer.");

		l.logistic = sigmoid;
		l.param_offset = param_idx;
		param_idx += (arr[i-1]+1)*arr[i];
		xavier_init(&n.params[param_idx], arr[i-1], arr[i]);
		printf("%f\n", n.params[param_idx]);
	
		n.layers[i-1] = l;
	}

	n.gpu_params = clCreateBuffer(SIEKNET_GLOBAL_CONTEXT, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * n.num_params, n.params, &err);
	check_error(err, "creating & copying gpu params");

	n.param_grad = clCreateBuffer(SIEKNET_GLOBAL_CONTEXT, CL_MEM_READ_WRITE, sizeof(float) * n.num_params, NULL, &err);
	check_error(err, "creating gpu param grads");

	n.network_input = clCreateBuffer(SIEKNET_GLOBAL_CONTEXT, CL_MEM_READ_WRITE, sizeof(float) * n.input_dimension, NULL, &err);
	check_error(err, "creating temp input buffer");

	n.output = ALLOCATE(float, n.output_dimension);
	return n;
}

void gpu_mlp_layer_forward(MLP_layer *l, cl_mem input, cl_mem params){
	printf("setting arg0\n");
	cl_mem z = l->z;
	check_error(clSetKernelArg(linear, 0, sizeof(cl_mem), &input), "arg0");
	printf("setting arg1\n");
	check_error(clSetKernelArg(linear, 1, sizeof(cl_mem), &z), "arg1");
	printf("setting arg2\n");
	check_error(clSetKernelArg(linear, 2, sizeof(cl_mem), &params), "arg2");
	printf("setting arg3\n");
	check_error(clSetKernelArg(linear, 3, sizeof(int), &l->size), "arg3");
	printf("setting arg4\n");
	clSetKernelArg(linear, 4, sizeof(int), &l->param_offset);

	printf("setting arg 0\n");
	clSetKernelArg(l->logistic, 0, sizeof(cl_mem), &l->z);
	printf("setting arg 1\n");
	clSetKernelArg(l->logistic, 1, sizeof(cl_mem), &l->output);

	printf("enqueueing linear\n");
	check_error(clEnqueueNDRangeKernel(SIEKNET_GLOBAL_QUEUE, linear, 1, NULL, &l->size, NULL, 0, NULL, NULL), "couldn't enqueue linear kernel");
	printf("enqueueing logistic\n");
	check_error(clEnqueueNDRangeKernel(SIEKNET_GLOBAL_QUEUE, l->logistic, 1, NULL, &l->size, NULL, 0, NULL, NULL), "couldn't enqueue logistic kernel");

}

void gpu_mlp_forward(MLP *n, float *x){
	printf("copying input\n");
	check_error(clEnqueueWriteBuffer(SIEKNET_GLOBAL_QUEUE, n->network_input, 1, 0, sizeof(float) * n->input_dimension, x, 0, NULL, NULL), "enqueuing network input");

	cl_mem input = n->network_input;
	for(int i = 0; i < n->depth; i++){
		printf("doing layer %d\n", i+1);
		MLP_layer *l = &n->layers[i];
		gpu_mlp_layer_forward(l, input, n->gpu_params);
		input = l->output;
	}
	
	clEnqueueReadBuffer(SIEKNET_GLOBAL_QUEUE, input, 1, 0, sizeof(float) * n->output_dimension, n->output, 0, NULL, NULL);
	n->guess = 0;
	for(int i = 0; i < n->output_dimension; i++){
		printf("%f\n", n->output[i]);
		if(n->output[n->guess] < n->output[i])
			n->guess = i;
	}
	
}

void gpu_mlp_backward(MLP *n){

}

#endif

MLP mlp_from_arr(size_t arr[], size_t size){
#ifndef GPU
	return cpu_mlp_from_arr(arr, size);
#else
	return gpu_mlp_from_arr(arr, size);
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
void save_mlp(const MLP *n, const char* filename){
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
		fprintf(fp, "%f", n->params[i]);
		if(i < n->num_params-1) fprintf(fp, " ");
		else fprintf(fp, "\n");
	}
	fclose(fp);
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
	n = mlp_from_arr(arr, num_layers+1);
	for(int i = 0; i < n.num_params; i++){
		f = fscanf(fp, "%f", &n.params[i]);
	}
	return n;
}
