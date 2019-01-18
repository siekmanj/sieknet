/* Author: Jonah Siekmann
 * Written 7/24/2018, updated 1/17/2019
 * This is a multilayer perceptron implementation. I've tested it with mnist and a few trivial problems.
 * Every function beginning with static is meant for internal use only. You may call any other function.
 */

#include <mlp.h>
#include <math.h>
#include <string.h>

#define PRINTLIST(name, len) printf("printing %s: [", #name); for(int xyz = 0; xyz < len; xyz++){printf("%5.4f", name[xyz]); if(xyz < len-1) printf(", "); else printf("]\n");}

/*
 * A function for calculating the inner product of two vectors.
 */
static float inner_product(const float *x, const float *y, size_t length){
	float sum = 0;
	for(long i = 0; i < length; i++){
		sum += x[i] * y[i];	
	}
	return sum;
}


/*
 * Description: Calculates the activations of a layer with sigmoid.
 * layerptr: A pointer to the layer for which outputs will be calculated.
 */
void sigmoid(MLP_layer* layer){
  for(int i = 0; i < layer->size; i++){
    layer->output[i] = (1 / (1 + exp(-layer->neurons[i].input)));
  }
}

/*
 * Description: Calculates the activations of a layer using ReLu.
 * layerptr: A pointer to the layer for which outputs will be calculated.
 * warning: this does not appear to be stable.
 */
void relu(MLP_layer* layer){
	for(int i = 0; i < layer->size; i++){
		float x = layer->neurons[i].input;
		layer->output[i] = x;
	}
}

/*
 * Description: Calculates the activations of a layer using tanh.
 * layerptr: A pointer to the layer for which outputs will be calculated.
 */
 void hypertan(MLP_layer* layer){
	for(int i = 0; i < layer->size; i++){
		float x = layer->neurons[i].input;
		layer->output[i] = ((exp(x) - exp(-x))/(exp(x) + exp(-x)));
	}
 }

float quadratic_cost(MLP *n, float *y){
  float sum = 0;
	float *o = n->output;
	printf("SUM:\n");
  for(int i = 0; i < n->output_dimension; i++){
		n->cost_gradient[i] = (2 * (y[i] - o[i]));
		sum += (y[i]-o[i]) * (y[i]-o[i]);
		printf("(%f - %f)^2 = %f\n", y[i], o[i], (y[i]-o[i]) * (y[i]-o[i]));
  }
	printf("COST: created gradients: ");
	PRINTLIST(n->cost_gradient, n->output_dimension);
  return sum;
}

float cross_entropy_cost(MLP *n, float *y){
  float sum = 0;
	float *o = n->output;
  for(int i = 0; i < n->output_dimension; i++){
		if(o[i] > 0.9999) o[i] = 0.9999;
		if(o[i] < 0.0001) o[i] = 0.0001;
		n->cost_gradient[i] = y[i]/o[i] + (1-y[i])/(1-o[i]);
		sum += -(y[i] * log(o[i]) + (1-y[i]) * log(1-o[i]));
  }
	printf("COST: %5.4f\n", sum);
  return sum;
}

/*
 * Description: Calculates the activation of a given neuron using softmax, and
 *              sets the partial derivative of the cost with respect to the activation.
 * layerptr: A pointer to the layer for which outputs will be calculated.
 */
void softmax(MLP_layer* layer){
  double sum = 0;

  for(int i = 0; i < layer->size; i++)
    sum += exp(layer->neurons[i].input);

  for(int i = 0; i < layer->size; i++)
    layer->output[i] = exp(layer->neurons[i].input) / sum;
}

/* 
 * Description: Creates a layer object
 */
static MLP_layer *create_MLP_layer(size_t input_dimension, size_t num_neurons, float *params, void(*logistic)(MLP_layer *layer)){
  MLP_layer *layer = (MLP_layer*)malloc(sizeof(MLP_layer));

  //Allocate every neuron in the layer
  Neuron* neurons = (Neuron*)malloc(num_neurons*sizeof(Neuron));

	int param_bound = num_neurons * input_dimension; //The number of parameters to read from network's param array
	int param_idx = 0;
  for(int i = 0; i < num_neurons; i++){
		neurons[i].bias = &params[param_idx];
		neurons[i].weights = &params[param_idx+1];
		param_idx += input_dimension + 1;
  }
	layer->input = NULL;
	layer->output = (float*)malloc(num_neurons*sizeof(float)); //allocate for layer outputs (forward pass)
	layer->gradient = (float*)malloc(input_dimension*sizeof(float)); //allocate for layer gradients (backward pass)

  layer->neurons = neurons;
  layer->size = num_neurons;
  layer->input_dimension = input_dimension;
  
	layer->logistic = logistic; //Set layer activation to sigmoid by default
  return layer;
}

/*
 * Description: a function called through a macro that allows creation of a network with any arbitrary number of layers.
 * arr: The array containing the sizes of each layer, for instance {28*28, 16, 10}.
 * size: The size of the array.
 */
MLP mlp_from_arr(size_t arr[], size_t size){
	MLP n;
	n.learning_rate = 0.05;
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
	//n.cost = cross_entropy_cost;
	n.cost = quadratic_cost;

	for(int i = 0; i < num_params; i++)
		n.params[i] = ((float)(rand()%10000)/50000 - .25);


 
	int param_idx = 0;

	for(int i = 1; i < size; i++){

			MLP_layer *l;
			size_t layer_size = arr[i];
			size_t input_dimension = arr[i-1];

			float *param_addr = &n.params[param_idx];

			param_idx += layer_size * (input_dimension+1);

			if(i < size-1)
				l = create_MLP_layer(input_dimension, layer_size, param_addr, sigmoid);
			else
				l = create_MLP_layer(input_dimension, layer_size, param_addr, softmax);
			
			n.layers[i-1] = *l;
	}
	n.output = n.layers[n.depth-1].output;
  return n;
}

static void mlp_layer_forward(MLP_layer *l, float *x){
	l->input = x; //need to save pointer for backward pass
	for(int i = 0; i < l->size; i++){
		float *w = l->neurons[i].weights;
		l->neurons[i].input = inner_product(x, w, l->input_dimension) + *l->neurons[i].bias;
	}
	l->logistic(l);
	PRINTLIST(l->output, l->size);
}

void mlp_forward(MLP *n, float *input){
	//getchar();
	float *x = input;
	for(int i = 0; i < n->depth; i++){
		printf("forward pass for layer %d:\n	", i);
		MLP_layer *l = &n->layers[i];
		mlp_layer_forward(l, x);
		x = l->output;
	}
	n->guess = 0;
	for(int i = 0; i < n->output_dimension; i++)
		if(n->output[n->guess] < n->output[i])
			n->guess = i;

	printf("	guess: %d (%f)\n", n->guess, n->output[n->guess]);
	
}

static void mlp_layer_backward(MLP_layer *l, float *grads, float learning_rate){
	/*
		for(int i = 0; i < input_layer->size; i++){
			float sum = 0;
			for(int j = 0; j < current->size; j++){
				float Wij = current->neurons[j].weights[i];
				float dActivation = current->neurons[j].dActivation;
				float gradient = current->neurons[j].gradient;
				sum += Wij * dActivation * gradient;
			}
			input_layer->neurons[i].gradient = sum;
		}
		*/
	for(int i = 0; i < l->input_dimension; i++){ //Zero gradients for previous layer
		float sum = 0;
		for(int j = 0; j < l->size; j++){
			float w = l->neurons[j].weights[i];
			float d = l->output[j] * (1 - l->output[j]);
			float g = grads[j];
			sum += w * d * g;
		}
		l->gradient[i] = sum;
	}

	for(int i = 0; i < l->size; i++){
		float gradient = grads[i]; //gradient of this neuron's output with respect to cost

		float d_output = l->output[i] * (1 - l->output[i]);
		//if(l->logistic == hypertan)
		//	d_output = 1 - l->output[i]*l->output[i]; //dTanh
		//else
		//	d_output = l->output[i] * (1 - l->output[i]); //dSigmoid and dSoftmax

		for(int j = 0; j < l->input_dimension; j++){
			float w = l->neurons[i].weights[j];
			float x = l->input[j];

			//l->gradient[j] 					 += gradient * d_output * w;
			l->neurons[i].weights[j] += gradient * d_output * x * learning_rate;
		}
		*l->neurons[i].bias += gradient * d_output * learning_rate;
	}
}

void mlp_backward(MLP *n){
	float *grads = n->cost_gradient;
	for(int i = n->depth-1; i >= 0; i--){
		printf("backward pass for layer %d\n	passing: ", i);
		PRINTLIST(grads, n->layers[i].size);
		mlp_layer_backward(&n->layers[i], grads, n->learning_rate);
		grads = n->layers[i].gradient;
		printf("	after: ");
		PRINTLIST(grads, n->layers[i].input_dimension);
	}
	printf("	backward pass concluded.\nparams: ");
	PRINTLIST(n->params, 10);
	//getchar();
}

/* 
 * Description: Calculates the quadratic cost for an output neuron.
 *               Also sets activation gradients for output layer.
 * neuron: The neuron for which cost will be calculated.
 * y: The expected value of the neuron
 *
static float quadratic_cost(Neuron *neuron, float y){
  neuron->gradient = (2 * (y-neuron->activation));
  return ((y-neuron->activation)*(y-neuron->activation));
}

/* 
 * Description: Calculates the cross entropy cost for an output neuron.
 *               Also sets activation gradients for output layer.
 * neuron: The neuron for which cost will be calculated.
 * y: The expected value of the neuron
 *
static float cross_entropy_cost(Neuron *neuron, float y){
  //Make sure we don't get divide by zero errors for safety
  if(neuron->activation < 0.00001) neuron->activation = 0.00001;
  else if(neuron->activation > 0.9999) neuron->activation = 0.9999;

  neuron->gradient = ((float)y)/neuron->activation - (float)(1-y)/(1.0-neuron->activation);

  float c = -(y * log(neuron->activation) + (1 - y) * log(1 - neuron->activation));
  if(isnan(c)){
    printf("NAN ALERT COST: %d, %f, %f, %f, %f\n", y, neuron->activation, log(neuron->activation), log(1 - neuron->activation), c);
    while(1);
  }
  return c;
}

/* Description: Calculates the cost of the output layer and sets the activation gradients for the output neurons.
 * output_layer: the last layer in the network.
 * label: the expected value chosen by the network.
 *
static float cost(MLP_layer *output_layer, float *expected){
  float sum = 0;
  for(int i = 0; i < output_layer->size; i++){
    int y = expected[i];

    Neuron *neuron = &output_layer->neurons[i];
	  //Calculate the cost from the desired value and actual neuron output
		sum += cross_entropy_cost(neuron, y);
		//sum += quadratic_cost(neuron, y);
  }
  return sum;
}*/

/*
static void propagate_gradients(MLP_layer *output_layer){
	MLP_layer *current = output_layer;
	while(current->input_layer != NULL){
		MLP_layer* input_layer = current->input_layer;

		for(int i = 0; i < input_layer->size; i++){
			float sum = 0;
			for(int j = 0; j < current->size; j++){
				float Wij = current->neurons[j].weights[i];
				float dActivation = current->neurons[j].dActivation;
				float gradient = current->neurons[j].gradient;
				sum += Wij * dActivation * gradient;
			}
			input_layer->neurons[i].gradient = sum;
		}
		current = input_layer;
	}
}
*/
/* 
 * Description: Performs backpropagation algorithm on the network.
 * output_layer: The last layer in the network.
 * label: The neuron that should have fired in the output layer.
 * learning_rate: The learning rate of the network.
 *
float backpropagate(MLP *n, float *expected){
  float c = cost(n->output, expected); //Calculate cost & set activation gradients in output layer
	propagate_gradients(n->output); //Calculate gradients for every other neuron in the network

  MLP_layer *current = n->output;
  while(current->input_layer != NULL){
    MLP_layer* input_layer = current->input_layer;

    for(int i = 0; i < current->size; i++){
      Neuron *currentNeuron = &current->neurons[i];
      float dActivation = currentNeuron->dActivation;
      float gradient = currentNeuron->gradient;

      //Calculate weight nudges
      for(int j = 0; j < input_layer->size; j++){
        float a = input_layer->neurons[j].activation;
        currentNeuron->weights[j] += a * dActivation * gradient * n->learning_rate;
      }
      //Calculate bias nudge
      currentNeuron->bias += dActivation * gradient * n->learning_rate;
    }
    current = current->input_layer;
  }
  return c;
}


/* 
 * Description: Calculates the outputs of each neuron in a layer based on the outputs & weights of the neurons in the preceding layer
 * layer: the layer for which outputs will be calculated from inputs of the previous layer.
 *
void calculate_inputs(MLP_layer *l){
  MLP_layer *input_layer = l->input_layer;
  for(int i = 0; i < l->size; i++){
		l->output[i] = inner_product(input_layer->output	
    float sum = 0;
    Neuron *current = &(layer->neurons[i]);
    for(int k = 0; k < input_layer->size; k++){
      sum += input_layer->neurons[k].activation * layer->neurons[i].weights[k];
      if(isnan(sum)){
        printf("NAN DURING OUTPUT CALC: %f, %f, %f\n", input_layer->neurons[k].activation, layer->neurons[i].weights[k], sum);
        while(1);
      }
    }
    current->input = sum + current->bias;
    if(isnan(current->input)){
      printf("NAN DURING INPUT ASSIGN: %f, %f\n", sum, current->bias);
      while(1);
    }
  }
}

/* 
 * Description: Does feed-forward, cost, and then backpropagation
 *
float mlp_step(MLP *n, float *x, float *y){
  feedforward(n, x);
  return backpropagate(n, y);
}


/*
 * Description: Performs the feed-forward operation on the network.
 * n: A pointer to the network.
 *
void mlp_forward(MLP *n, float *arr){
	set_outputs(n->input, arr);
  MLP_layer *current = n->input->output_layer;
  while(current != NULL){
    calculate_inputs(current); //Calculate inputs from previous layer
  	current->squish(current); //Squish current layer's inputs into non-linearity
    current = current->output_layer; //Advance to the next layer
  }
}

/*
 * Description: Deallocates a network's memory from the heap
 * n: the pointer to the output layer of the MLP
 *
void dealloc_network(MLP *n){
	int counter = 0;
	MLP_layer* current = n->output;
	while(current != NULL){
		for(int i = 0; i < current->size; i++){
			free(current->neurons[i].weights);
		}
		counter++;
		free(current->neurons);
		MLP_layer* temp = current->input_layer;
		free(current);
		current = temp;
	}
	n->input = NULL;
	n->output = NULL;
}


 /*
  * IO FUNCTIONS FOR READING AND WRITING TO A FILE
  */

static void writeToFile(FILE *fp, char *ptr){
  fprintf(fp, "%s", ptr);
  memset(ptr, '\0', strlen(ptr));
}

static void getWord(FILE *fp, char* dest){
  memset(dest, '\0', strlen(dest));
  //printf("bytes read: %lu\n", fread(dest, 1024, 1, fp));
  int res = fscanf(fp, " %1023s", dest);
}
/* 
 * Description: Saves the network's state to a file that can be read later.
 * n: A pointer to the network.
 * filename: The desired filename and path.
 *
void saveMLPToFile(MLP *n, char* filename){
 FILE *fp;
 char buff[1024];
 memset(buff, '\0', 1024);

 //Create file
 fp = fopen(filename, "w");
 printf("Saving to: %s\n", filename);
 memset(buff, '\0', strlen(buff));

 //Get network dimensions
 size_t size = 0;
 MLP_layer *current = n->input;
 while(1){
   if(current != NULL) size++;
   if(current == n->output) break;
   current = current->output_layer;
 }

 //Write header info to file
 strcat(buff, "MLP ");
 writeToFile(fp, buff); //Write identifier
 snprintf(buff, 100, "%lu ", size); //Convert num of layers to int
 writeToFile(fp, buff); //Write number of layers to file

 current = n->input;
 for(int i = 0; i < size; i++){
   //Write layer info to file
   strcat(buff, "layer ");
   writeToFile(fp, buff);
   snprintf(buff, 100, "%lu ", current->size);
   writeToFile(fp, buff);

   //Write neuron info to file
   if(current == n->input){
     strcat(buff, "INPUTLAYER ");
     writeToFile(fp, buff);
   }else{
     for(int j = 0; j < current->size; j++){
       MLP_layer* input_layer = current->input_layer;

       strcat(buff, "neuron ");
       writeToFile(fp, buff);
       snprintf(buff, 100, "%lu ", input_layer->size);
       writeToFile(fp, buff);

       for(int k = 0; k < input_layer->size; k++){
         snprintf(buff, 100, "%f ", current->neurons[j].weights[k]);
         writeToFile(fp, buff);
       }
       snprintf(buff, 100, "%f ", current->neurons[j].bias);
       writeToFile(fp, buff);
     }
   }
   current = current->output_layer;
 }
 fclose(fp);
}

/*
 * Description: Loads a network from a file.
 * filename: The path to the file.
 *
MLP loadMLPFromFile(const char *filename){
  FILE *fp = fopen(filename, "rb");
  char buff[1024];
  memset(buff, '\0', 1024);

  MLP n = initMLP();
  getWord(fp, buff); //Get first word to check if MLP file

  if(strcmp(buff, "MLP") != 0){
    printf("ERROR: [%s] is not MLP.\n", buff);
    return n;
  }

  //Get number of layers in network
  getWord(fp, buff);
  size_t size = strtol(buff, NULL, 10);

  for(int i = 0; i < size; i++){
    getWord(fp, buff);
    if(strcmp(buff, "layer") != 0){
      printf("PARSE ERROR\n");
      return n;
    }
    getWord(fp, buff);
    size_t layer_size = strtol(buff, NULL, 10);
    addMLP_layer(&n, layer_size);

    for(int j = 0; j < layer_size; j++){
      getWord(fp, buff);
      if(strcmp(buff, "INPUTLAYER") == 0) {
        break;
      }
      getWord(fp, buff);
      size_t number_of_weights = strtol(buff, NULL, 10);
      MLP_layer *input_layer = n.output->input_layer;
      for(int k = 0; k < number_of_weights; k++){
        getWord(fp, buff);
        float weight = strtod(buff, NULL);
        n.output->neurons[j].weights[k] = weight;
      }
      getWord(fp, buff);
      float bias = strtod(buff, NULL);
      n.output->neurons[j].bias = bias;
    }
  }
  fclose(fp);
  return n;
}


/*
 * Functions for debugging
 *
void printWeights(MLP_layer *layer){
  MLP_layer *previousMLP_layer = layer->input_layer;
	for(int i = 0; i < layer->size; i++){
		printf("Neuron %d: ", i);
	}
	printf("\n");
	for(int i = 0; i < previousMLP_layer->size; i++){
		for(int j = 0; j < layer->size; j++){
			if(j > 9) printf(" %9.2f ", layer->neurons[j].weights[i]);
			else printf("%8.2f  ", layer->neurons[j].weights[i]);
		}
		printf("\n");
	}
	printf("\n");
	for(int i = 0; i < layer->size; i++){
			if(i > 9) printf(" %9.2f ", layer->neurons[i].bias);
			else printf("%8.2f  ", layer->neurons[i].bias);
	}
	printf("\n");
}

void printActivationGradients(MLP_layer *layer){
  printf("activation gradients:\n");
  for(int i = 0; i < layer->size; i++){
    printf("  Neuron %d: %f from %f\n", i, layer->neurons[i].gradient, layer->neurons[i].activation);
  }
}
void printOutputs(MLP_layer *layer){
  for(int i = 0; i < layer->size; i++){
    float val = layer->neurons[i].activation;
		if(i > 9) printf(" %9.4f ", val);
		else printf("%8.4f  ", val);
		}
		printf("\n");
}
void prettyprint(MLP_layer *layer){
  for(int i = 0; i < layer->size; i++){
    float val = layer->neurons[i].activation;
    if(!(i % (int)sqrt(layer->size))) printf("\n");
		if(val <= 0.5) printf(".");
    if(val > 0.5) printf("A");
		}
    printf("\n");
}
*/
