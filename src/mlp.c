/* Author: Jonah Siekmann
 * Written 7/24/2018, updated 1/17/2019
 * This is a multilayer perceptron implementation. I've tested it with mnist and a few trivial problems.
 * Every function beginning with static is meant for internal use only. You may call any other function.
 */

#include <mlp.h>
#include <math.h>
#include <string.h>

#define PRINTLIST(name, len) printf("printing %s: [", #name); for(int xyz = 0; xyz < len; xyz++){printf("%5.4f", name[xyz]); if(xyz < len-1) printf(", "); else printf("]\n");}

#define MAX_GRAD 100
#define DEBUG 0

/*
 * Calculates the inner product of two vectors.
 */
static float inner_product(const float *x, const float *y, size_t length){
	float sum = 0;
	for(long i = 0; i < length; i++){
		sum += x[i] * y[i];	
	}
	return sum;
}


/*
 * Calculates the activations of a layer with sigmoid.
 */
void sigmoid(MLP_layer* layer){
  for(int i = 0; i < layer->size; i++){
    layer->output[i] = (1 / (1 + exp(-layer->neurons[i].input)));
  }
}

/*
 * Calculates the activations of a layer using ReLu.
 */
void relu(MLP_layer* layer){
	for(int i = 0; i < layer->size; i++){
		float x = layer->neurons[i].input;
		layer->output[i] = x;
	}
}

/*
 * Calculates the activations of a layer using tanh.
 */
 void hypertan(MLP_layer* layer){
	for(int i = 0; i < layer->size; i++){
		float x = layer->neurons[i].input;
		if(x > 7.0) layer->output[i] = 0.999998;
		else if(x < -7.0) layer->output[i] = -0.999998;
		else layer->output[i] = ((exp(x) - exp(-x))/(exp(x) + exp(-x)));
	}
 }

/*
 * Calculates the gradients wrt cost function given a label vector y.
 */
float quadratic_cost(MLP *n, float *y){
  float sum = 0;
	float *o = n->output;
  for(int i = 0; i < n->output_dimension; i++){
		n->cost_gradient[i] = (y[i] - o[i]);
		sum += 0.5*(y[i]-o[i]) * (y[i]-o[i]);
  }
  return sum;
}

float cross_entropy_cost(MLP *n, float *y){
  float sum = 0;
	float *o = n->output;
  for(int i = 0; i < n->output_dimension; i++){
		if(o[i] > 0.9999) o[i] = 0.9999;
		if(o[i] < 0.0001) o[i] = 0.0001;
		float grad = (y[i]/o[i]) - ((1-y[i])/(1-o[i]));
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
		n->cost_gradient[i] = grad;
		sum += -(y[i] * log(o[i]) + (1-y[i]) * log(1-o[i]));
  }
  return sum;
}

/*
 * Calculates the activation of a given neuron using softmax.
 */
void softmax(MLP_layer* layer){
  double sum = 0;

  for(int i = 0; i < layer->size; i++)
    sum += exp(layer->neurons[i].input);

  for(int i = 0; i < layer->size; i++)
    layer->output[i] = exp(layer->neurons[i].input) / sum;
}

/* 
 * Creates a layer object
 */
MLP_layer create_MLP_layer(size_t input_dimension, size_t num_neurons, float *params, void(*logistic)(MLP_layer *layer)){
  MLP_layer layer;

  //Allocate every neuron in the layer
  Neuron* neurons = (Neuron*)malloc(num_neurons*sizeof(Neuron));

	int param_bound = num_neurons * input_dimension; //The number of parameters to read from network's param array
	int param_idx = 0;
  for(int i = 0; i < num_neurons; i++){
		neurons[i].bias = &params[param_idx];
		neurons[i].weights = &params[param_idx+1];
		param_idx += input_dimension + 1;
		
		//Xavier (or Xavier-like) initialization
		for(int j = 0; j < input_dimension; j++){ 
			float rand_weight = (((float)rand())/((float)RAND_MAX)) * sqrt(2.0 / (input_dimension + num_neurons));
			if(rand()&1) rand_weight *= -1;
			neurons[i].weights[j] = rand_weight;
		}
		float rand_bias = (((float)rand())/((float)RAND_MAX)) * sqrt(2.0 / (input_dimension + num_neurons));
		if(rand()&1) rand_bias *= -1;
		*neurons[i].bias = rand_bias;
  }
	layer.input = NULL; //set in forward pass
	layer.output = (float*)malloc(num_neurons*sizeof(float)); //allocate for layer outputs (forward pass)
	layer.gradient = (float*)malloc(input_dimension*sizeof(float)); //allocate for layer gradients (backward pass)

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
	n.learning_rate = 0.1;
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
	n.params = (float*)malloc(num_params*sizeof(float)); //contains weights and biases of every layer
	n.cost_gradient = (float*)malloc(n.output_dimension * sizeof(float));
	n.layers = (MLP_layer*)malloc((size-1)*sizeof(MLP_layer));
	n.cost = cross_entropy_cost;

	int param_idx = 0;
	for(int i = 1; i < size; i++){

			MLP_layer l;
			size_t layer_size = arr[i];
			size_t input_dimension = arr[i-1];

			float *param_addr = &n.params[param_idx];

			param_idx += layer_size * (input_dimension+1);

			if(i < size-1)
				l = create_MLP_layer(input_dimension, layer_size, param_addr, sigmoid);
			else
				l = create_MLP_layer(input_dimension, layer_size, param_addr, softmax);
			
			n.layers[i-1] = l;
	}
	n.output = n.layers[n.depth-1].output;
  return n;
}

/*
 * Does a forward pass for a single layer.
 */
void mlp_layer_forward(MLP_layer *l, float *x){
	l->input = x; //need to save pointer for backward pass
	for(int i = 0; i < l->size; i++){
		float *w = l->neurons[i].weights; 
		l->neurons[i].input = inner_product(x, w, l->input_dimension) + *l->neurons[i].bias;
	}
	l->logistic(l); //Apply this layer's logistic function
}

/*
 * Does a forward pass for the entire network.
 */
void mlp_forward(MLP *n, float *input){
	float *x = input;
	for(int i = 0; i < n->depth; i++){
		MLP_layer *l = &n->layers[i];
		mlp_layer_forward(l, x); //Do forward pass for this layer
		x = l->output; //Use this layer's output as the next layer's input.
	}
	n->guess = 0;
	for(int i = 0; i < n->output_dimension; i++)
		if(n->output[n->guess] < n->output[i])
			n->guess = i;
}

/*
 * Calculates logistic function derivatives in terms of logistic output
 */
float differentiate(const float x, const void (*logistic)(MLP_layer*)){
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
 * Propagates gradients throughout network using the chain rule (does not do parameter update)
 */
void propagate_gradients(MLP *n, float *gradient){
	float *grads = gradient;
	for(int i = n->depth-1; i >= 0; i--){
		MLP_layer *l = &n->layers[i];
		for(int j = 0; j < l->input_dimension; j++){
			float sum = 0;
			for(int k = 0; k < l->size; k++){
				float w = l->neurons[k].weights[j];
				float d = differentiate(l->output[k], l->logistic);
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
void mlp_layer_backward(MLP_layer *l, float *grads, float learning_rate){
	for(int i = 0; i < l->size; i++){
		float gradient = grads[i]; //gradient of this neuron's output with respect to cost
		float d_output = differentiate(l->output[i], l->logistic);

		for(int j = 0; j < l->input_dimension; j++){
			float x = l->input[j];
			l->neurons[i].weights[j] += gradient * d_output * x * learning_rate;
		}
		*l->neurons[i].bias += gradient * d_output * learning_rate;
	}
}

/*
 * Does backward pass for entire network (does paramter update)
 */
void mlp_backward(MLP *n){
	float *grads = n->cost_gradient;
	propagate_gradients(n, grads);
	for(int i = n->depth-1; i >= 0; i--){
		mlp_layer_backward(&n->layers[i], grads, n->learning_rate);
		grads = n->layers[i].gradient;
	}
}

/*
 * Deallocates a network's memory from the heap
 */
void dealloc_network(MLP *n){
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
  FILE *fp = fopen(filename, "rb");
  char buff[1024];
  memset(buff, '\0', 1024);

  getWord(fp, buff); //Get first word to check if MLP file

  if(strcmp(buff, "MLP") != 0){
    printf("ERROR: [%s] is not MLP.\n", buff);
    exit(1);
  }
	size_t num_layers, input_dim;

	fscanf(fp, "%lu %lu", &num_layers, &input_dim);
	size_t arr[num_layers+1];
	arr[0] = input_dim;
	for(int i = 1; i <= num_layers; i++){
		fscanf(fp, " %lu", &arr[i]);
	}

	MLP n;
	n = mlp_from_arr(arr, num_layers+1);
	for(int i = 0; i < n.num_params; i++){
		fscanf(fp, "%f", &n.params[i]);
	}
  return n;
}
