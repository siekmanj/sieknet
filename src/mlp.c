/* Author: Jonah Siekmann
 * Written 7/24/2018, updated 1/17/2019
 * This is a multilayer perceptron implementation. I've tested it with mnist and a few trivial problems.
 * Every function beginning with static is meant for internal use only. You may call any other function.
 */

#include <mlp.h>
#include <math.h>
#include <string.h>

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
 * Calculates the activations of a layer with sigmoid.
 */
void sigmoid(const float *z, float *dest, size_t dim){
	for(int i = 0; i < dim; i++){
		dest[i] = (1 / (1 + exp(-z[i])));
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

float mlp_cost(MLP *n, float *y){
	return n->cost_fn(n->output, y, n->cost_gradient/*[n->b++]*/, n->output_dimension);
}

/* 
 * Creates a layer object
 */
MLP_layer create_MLP_layer(size_t input_dimension, size_t num_neurons, float *params, float *param_grad, void(*logistic)(const float *, float *, size_t)){
	MLP_layer layer;

	//Allocate every neuron in the layer
	//Neuron* neurons = (Neuron*)malloc(num_neurons*sizeof(Neuron));
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
	//layer.input = NULL; //set in forward pass
//	layer.input = ALLOCATE(float*, MAX_BATCH_SIZE);
	//layer.output = (float*)malloc(MAX_BATCH_SIZE*sizeof(float)); //allocate for layer outputs (forward pass)

	layer.z = ALLOCATE(float, num_neurons);
	layer.output = ALLOCATE(float, num_neurons);
	/*
	layer.output = ALLOCATE(float*, MAX_BATCH_SIZE);
	for(int i = 0; i < MAX_BATCH_SIZE; i++)
		layer.output[i] = ALLOCATE(float, num_neurons);
	*/
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
MLP mlp_from_arr(size_t arr[], size_t size){
	MLP n;
	//n.learning_rate = 0.1;
	n.input_dimension = arr[0];
	n.output_dimension = arr[size-1];
	n.depth = size-1;
	//n.batch_size = 5;
	//n.b = 0;

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
	/*
	n.cost_gradient = ALLOCATE(float*, MAX_BATCH_SIZE);
	for(int i = 0; i < MAX_BATCH_SIZE; i++)
		n.cost_gradient[i] = ALLOCATE(float, n.output_dimension);

	n.network_input = ALLOCATE(float*, MAX_BATCH_SIZE);
	for(int i = 0; i < MAX_BATCH_SIZE; i++)
		n.network_input[i] = ALLOCATE(float, n.input_dimension);
	*/
	//n.layers = (MLP_layer*)malloc((size-1)*sizeof(MLP_layer));
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
				l = create_MLP_layer(input_dimension, layer_size, param_addr, grad_addr, sigmoid);
			else
				l = create_MLP_layer(input_dimension, layer_size, param_addr, grad_addr, softmax);
			
			n.layers[i-1] = l;
	}
	n.output = n.layers[n.depth-1].output/*[n.b]*/;
	return n;
}

/*
 * Does a forward pass for a single layer.
 */
void mlp_layer_forward(MLP_layer *l, float *x/*, size_t batch_idx*/){
	l->input = x; //need to save pointer for backward pass
	for(int i = 0; i < l->size; i++){
		float *w = l->neurons[i].weights; 
		l->z[i] = inner_product(x, w, l->input_dimension) + *l->neurons[i].bias;
	}
	l->logistic(l->z, l->output/*[batch_idx]*/, l->size); //Apply this layer's logistic function
}

/*
 * Does a forward pass for the entire network.
 */
void mlp_forward(MLP *n, float *input){
	//printf("doing forward!\n");
	//for(int i = 0; i < n->input_dimension; i++){
	//	n->network_input[n->b][i] = input[i];
	//}
	float *x = input;//n->network_input[n->b];
	for(int i = 0; i < n->depth; i++){
		//printf("doing batch idx %d for layer %d!\n", n->b, i);
		MLP_layer *l = &n->layers[i];
		mlp_layer_forward(l, x/*, n->b*/); //Do forward pass for this layer
		x = l->output/*[n->b]*/; //Use this layer's output as the next layer's input
		//printf("BATCH %d: CREATED SQUISHED OUTPUT FOR LAYER %d:\n", n->b, i);
		//PRINTLIST(x, l->size);
		//printf("did a layerfoward!\n");
	}
	n->guess = 0;
	for(int i = 0; i < n->output_dimension; i++)
		if(n->output[n->guess] < n->output[i])
			n->guess = i;
	//printf("did a forward!\n");
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
 * Handy function for zeroing out a 2d array
 */
void zero_2d_arr(float **arr, size_t sequence_length, size_t input_dimension){
	for(long i = 0; i < sequence_length; i++){
		for(long j = 0; j < input_dimension; j++){
			arr[i][j] = 0.0;
		}
	}
}

static void avg_2d_arr(float **arr, float *dest, size_t batches, size_t size){
	for(int i = 0; i < size; i++)
		dest[i] = 0.0;

	//printf("avg 2darr before avging:\n");
	//PRINTLIST(dest, size);
	for(int batch = 0; batch < batches; batch++){
		for(int i = 0; i < size; i++){
			dest[i] += arr[batch][i];
		}
	}
	for(int i = 0; i < size; i++){
		dest[i] /= (float)batches;
	}
	//printf("Made avg 2d arr of size %d:\n", size);
	//PRINTLIST(dest, size);

}

/*
 * Propagates gradients throughout network using the chain rule (does not do parameter update)
 */
void propagate_gradients(MLP *n, float *gradient/*, size_t batches*/){
	float *grads = gradient;
	for(int i = n->depth-1; i >= 0; i--){
		MLP_layer *l = &n->layers[i];
		
		//float *avg = l->output[MAX_BATCH_SIZE-1];
		//avg_2d_arr(l->output, avg, batches, l->size);
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
void mlp_layer_backward(MLP_layer *l, float *grads/*, float *avg_ins, float learning_rate*/){
	//float *avg_outs = l->output[MAX_BATCH_SIZE-1];
	float *avg_outs = l->output;
	for(int i = 0; i < l->size; i++){
		float gradient = grads[i]; //gradient of this neuron's output with respect to cost
		float d_output = differentiate(avg_outs[i], l->logistic);

		for(int j = 0; j < l->input_dimension; j++){
			//float x = avg_ins[j];
			float x = l->input[j];
			l->neurons[i].weight_grad[j] += gradient * d_output * x;// * learning_rate;
		}
		*l->neurons[i].bias_grad += gradient * d_output;// * learning_rate;
	}
}

/*
 * Does backward pass for entire network (does paramter update)
 */
void mlp_backward(MLP *n){
	//if(n->b < n->batch_size) return;

	//avg_2d_arr(n->cost_gradient, n->cost_gradient[MAX_BATCH_SIZE-1], n->batch_size, n->output_dimension);
	//avg_2d_arr(n->network_input, n->network_input[MAX_BATCH_SIZE-1], n->batch_size, n->input_dimension);

	float *grads = n->cost_gradient/*[MAX_BATCH_SIZE-1]*/;
	propagate_gradients(n, grads/*, n->batch_size*/);
	for(int i = n->depth-1; i >= 0; i--){
		//float *avg_ins = n->layers[i-1].output[MAX_BATCH_SIZE-1];
		mlp_layer_backward(&n->layers[i], grads/*, avg_ins, n->learning_rate*/);
		grads = n->layers[i].gradient;
	}
	//float *avg_ins = n->network_input[MAX_BATCH_SIZE-1];
	//mlp_layer_backward(&n->layers[0], grads, avg_ins, n->learning_rate);

	//n->b = 0;
	//zero_2d_arr(n->cost_gradient, MAX_BATCH_SIZE, n->output_dimension);
	//zero_2d_arr(n->network_input, MAX_BATCH_SIZE, n->input_dimension);
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
