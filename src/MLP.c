/* Author: Jonah Siekmann
 * Written 7/24/2018, updated 10/15/2018
 * This is a multilayer perceptron implementation using backpropagation. I've tested it with mnist and a few trivial problems.
 * It is intended to be a building block for more complex architectures. Currently I am attempting to build a genetic training algorithm.
 * Every function beginning with static is meant for internal use only. You may call any other function.
 */

#include "MLP.h"
#include <math.h>
#include <string.h>

#define MAX_LAYERS 20

/*
 * A function for calculating the inner product of two vectors.
 */
static float inner_product(float *x, float *y, size_t length){
	float sum = 0;
	for(long i = 0; i < length; i++){
		sum += x[i] * y[i];	
	}
	return sum;
}


/*
 * Description: Calculates the activation of a given neuron using sigmoid, and
 *              sets the partial derivative of the cost with respect to the activation.
 * layerptr: A pointer to the layer for which outputs will be calculated.
 */
void sigmoid(MLP_layer* layer){
  for(int i = 0; i < layer->size; i++){
    layer->output[i] = (1 / (1 + exp(-layer->neurons[i].input)));
  }
}

/*
 * Description: Calculates the activation of a given neuron using ReLu, and
 *              sets the partial derivative of the cost with respect to the 
 *							activation.
 * layerptr: A pointer to the layer for which outputs will be calculated.
 * warning: this does not appear to be stable with high learning rates.
 */
void relu(MLP_layer* layer){
	for(int i = 0; i < layer->size; i++){
		float x = layer->neurons[i].input;
		layer->output[i] = x;
	}
}
/*
 * Description: Calculates the activation of a given neuron using hyperbolic tangent, and
 *              sets the partial derivative of the cost with respect to the activation.
 * layerptr: A pointer to the layer for which outputs will be calculated.
 */
 void hypertan(MLP_layer* layer){
	for(int i = 0; i < layer->size; i++){
		float x = layer->neurons[i].input;
		layer->output[i] = ((exp(x) - exp(-x))/(exp(x) + exp(-x)));
//		layer->neurons[i].dActivation = set_output * (4/((exp(x) + exp(-x)) * (exp(x) + exp(-x))));
	}
 }

/*
 * Description: Calculates the activation of a given neuron using softmax, and
 *              sets the partial derivative of the cost with respect to the activation.
 * layerptr: A pointer to the layer for which outputs will be calculated.
 * NOTE: potentially stable (dActivation tends toward 0 with very large/very negative inputs, keep learning rate low)
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
MLP_layer *create_MLP_layer(size_t input_dimension, size_t num_neurons, float *params, void(*logistic)(MLP_layer *layer)){
	printf("inside create_layer!\n");
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

  layer->size = num_neurons;
  layer->neurons = neurons;
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
	n.cost = NULL;
	n.head = NULL;
	n.tail = NULL;
	n.learning_rate = 0.05;

	size_t num_params = 0;
	size_t num_outputs = 0;
	for(int i = 1; i < size; i++){
		num_params += (arr[i-1]+1)*arr[i];
		num_outputs += arr[i];
	}

	n.num_params = num_params;
	n.params = (float*)malloc(num_params*sizeof(float)); 

	n.activations = (float*)malloc(num_outputs*sizeof(float));

	int param_idx = 0;
	int activ_idx = 0;
	for(int i = 1; i < size; i++){
			size_t input_dimension = arr[i-1];
			size_t layer_size = arr[i];

//			float *param_addr = &n.params[i*(input_dimension+1)];
//			float *output_addr = &n.activations[i*la
			MLP_layer *l;

			if(i < size-1)
				l = create_MLP_layer(input_dimension, layer_size, param_addr, output_addr, sigmoid);
			else
				l = create_MLP_layer(input_dimension, layer_size, param_addr, output_addr, softmax);

			if(!n.head) n.head = l;
			if(!n.tail){
				l->input_layer = n.head;
				n.head->output_layer = l;
			}else{
				l->input_layer = n.tail;
				n.tail->output_layer = l;
			}
			n.tail = l;
	}
  return n;
}

void mlp_layer_forward(MLP_layer *l){
	//calculate linear inputs
	float x = l->input;
	for(int i = 0; i < l->size; i++)
		float w = l->neurons[i].weights;
		l->neurons[i].input = inner_product(x, w, l->input_dimension);
	}
	l->logistic(l);
}

void mlp_forward(MLP *n, float *x){
	MLP_layer *l = n->head;
	l->input = x;
	while(l){
		mlp_layer_forward(l);
		l = l->output_layer;
	}
}

void mlp_layer_backward(MLP_layer *l, float *grads){
	for(int i = 0; i < l->size; i++){
		
	}

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
