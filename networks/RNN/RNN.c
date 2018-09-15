
/* Author: Jonah Siekmann
 * 8/10/2018
 * This is an attempt at writing a recurrent neural network (RNN) Every function beginning with static is meant for internal use only. You may call any other function.
 */

#include "RNN.h"
#include <math.h>
#include <string.h>

/*
 * Description: Calculates the activation of a given neuron using sigmoid, and
 *              sets the partial derivative of the cost with respect to the activation.
 * layerptr: A pointer to the layer for which outputs will be calculated.
 *
static void sigmoid(void* layerptr){
  Layer* layer = (Layer*)layerptr;
  for(int i = 0; i < layer->size; i++){
    layer->neurons[i].activation = 1 / (1 + exp(-layer->neurons[i].input));
    layer->neurons[i].dActivation = layer->neurons[i].activation * (1 - layer->neurons[i].activation);
  }
}

/*
 * Description: Calculates the activation of a given neuron using softmax, and
 *              sets the partial derivative of the cost with respect to the activation.
 * layerptr: A pointer to the layer for which outputs will be calculated.
 * NOTE: potentially stable(?)
 *
static void softmax(void* layerptr){
  Layer* layer = (Layer*)layerptr;
  //Calculate denominator sum
  double sum = 0;
  double logC = 0;

  for(int i = 0; i < layer->size; i++){
    if(layer->neurons[i].input > logC) logC = layer->neurons[i].input;
    sum += exp(layer->neurons[i].input);
  }

  for(int i = 0; i < layer->size; i++){
    layer->neurons[i].activation = exp(layer->neurons[i].input) / sum;
    layer->neurons[i].dActivation = layer->neurons[i].activation * (1 - layer->neurons[i].activation);
  }
}

/* 
 * Description: Allocates an array of Neurons, and then allocates and initializes each Neuron's weights and biases.
 * size: the size of the layer currently being build
 * previousLayer: the preceding layer in the network (can be NULL).
 *
static Layer *create_layer(size_t size, Layer *previousLayer){
  Layer *layer = (Layer*)malloc(size*sizeof(Layer));
  size_t previous_layer_size = 0;
  if(previousLayer != NULL){
    previous_layer_size = previousLayer->size;
    previousLayer->output_layer = layer;
  }

  //Allocate every neuron in the layer
  Neuron* neurons = (Neuron*)malloc(size*sizeof(Neuron));
  for(int i = 0; i < size; i++){
    //Allocate weights
    float *weights = (float*)malloc(previous_layer_size*sizeof(float));
    for(int j = 0; j < previous_layer_size; j++){
      weights[j] = ((float)(rand()%70)-35)/10; //Slight bias towards positive weights
    }
    neurons[i].weights = weights;
    neurons[i].bias = ((float)(rand()%70)-35)/10;

  }
  layer->size = size;
  layer->neurons = neurons;
  layer->input_layer = previousLayer;
  layer->squish = sigmoid;
  return layer;
}

/* 
 * Description: Sets the outputs of a layer manually, instead of calculating them.
 *              The purpose of this function is to set the outputs of the input layer.
 * layer: The layer whose outputs will be set.
 * outputs: The array of outputs which will be assigned to the layer
 * NOTE: dimensions of outputs and layer must both be size*size
 *
static void set_outputs(Layer *layer, char *outputs){
  for(int i = 0; i < layer->size; i++){
    layer->neurons[i].activation = outputs[i];
  }
}

/* 
 * Description: Calculates the quadratic cost for an output neuron.
 *               Also sets activation gradients for output layer.
 * neuron: The neuron for which cost will be calculated.
 * y: Whether or not the neuron was supposed to fire or not (0 or 1)
 *
static float quadratic_cost(Neuron *neuron, int y){
  neuron->activationGradient = (2 * (y-neuron->activation));
  return ((y-neuron->activation)*(y-neuron->activation));
}

/* 
 * Description: Calculates the cross entropy cost for an output neuron.
 *               Also sets activation gradients for output layer.
 * neuron: The neuron for which cost will be calculated.
 * y: Whether or not the neuron was supposed to fire or not (0 or 1)
 *
static float cross_entropy_cost(Neuron *neuron, int y){
  //Make sure we don't get divide by zero errors for safety
  if(neuron->activation < 0.000001) neuron->activation = 0.000001;
  else if(neuron->activation > 0.99999) neuron->activation = 0.99999;

  neuron->activationGradient = ((float)y)/neuron->activation - (float)(1-y)/(1.0-neuron->activation);

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
static float cost(Layer *output_layer, int label){
  float sum = 0;
  for(int i = 0; i < output_layer->size; i++){
    int y = (i==label);

    Neuron *neuron = &output_layer->neurons[i];
	  //Calculate the cost from the desired value and actual neuron output
		sum += cross_entropy_cost(neuron, y);
		//sum += quadratic_cost(neuron, y);
  }
  return sum;
}

/* 
 * Description: Performs BPTT algorithm on the network.
 * output_layer: The last layer in the network.
 *
float backpropagate_through_time(Layer *output_layer, int label, float plasticity){
  float c = cost(output_layer, label); //Calculate cost & set activation gradients in output layer

  Layer *current = output_layer;
  while(current->input_layer != NULL){
    Layer* input_layer = (Layer*)(current->input_layer);
    for(int i = 0; i < input_layer->size; i++){
      //Calculate activation gradients in input layer BEFORE doing nudges to weights and biases in the current layer
      float sum = 0;
      for(int j = 0; j < current->size; j++){
        float dSig = current->neurons[j].dActivation;
        float weight = current->neurons[j].weights[i];
        float activationGradient = current->neurons[j].activationGradient;
        sum += weight*dSig*activationGradient;
        if(isnan(sum)){
          printf("NAN DURING BACKPROP: %f, %f, %f\n", dSig, weight, activationGradient);
          while(1);
        }
      }
      input_layer->neurons[i].activationGradient = sum*plasticity;
    }
    for(int i = 0; i < current->size; i++){
      Neuron *currentNeuron = &current->neurons[i];
      float dSig = currentNeuron->dActivation;
      float activationGradient = currentNeuron->activationGradient;

      //Calculate weight nudges
      for(int j = 0; j < input_layer->size; j++){
        float a = input_layer->neurons[j].activation;
        float in = input_layer->neurons[j].input;
        currentNeuron->weights[j] += a*dSig*activationGradient*plasticity;
      }
      //Calculate bias nudge
      currentNeuron->bias += dSig*activationGradient*plasticity;
    }
    current = current->input_layer;
  }
  return c;
}


/* 
 * Description: Calculates the outputs of each neuron in a layer based on the outputs & weights of the neurons in the preceding layer
 * layer1: the layer with existing outputs
 * layer2: the layer whose outputs will be calculated
 * important: dimensions of weights of layer2 must match dimensions of neurons of layer1
 *
static void calculate_outputs(Layer *layer){
  Layer *input_layer = (Layer*)(layer->input_layer);
  for(int i = 0; i < layer->size; i++){
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
  layer->squish(layer);
}

/* 
 * Description: Initializes an multilayer perceptron object.
 */
static RNN initRNN(){
  RNN n;
  n.input = NULL;
  n.output = NULL;
//	n.recurrent = NULL;
  n.performance = 0;
  n.plasticity = .25;
  return n;
}

/*
 * Description: a function called through a macro that allows creation of a network with any arbitrary number of layers.
 * arr: The array containing the sizes of each layer, for instance {28*28, 16, 10}.
 * size: The size of the array.
 */
RNN rnn_from_arr(size_t arr[], size_t size){
	RNN n = initRNN();
	size_t input_size = 0;
	for(int i = 0; i < size-1; i++){
		input_size += arr[i];
	}
	for(int i = 0; i < size; i++){
		if(i == 0) addLayer(&n, input_size);
		else addLayer(&n, arr[i]);
	}
  return n;
}

/* 
 * Description: Adds a layer to the RNN object. Layers will be inserted from input layer onward.
 * n: the pointer to the network.
 * size: the desired size of the layer to be added.
 *
void addLayer(RNN *n, size_t size){
  Layer *newLayer = create_layer(size, n->output);
  if(n->output != NULL) n->output->squish = sigmoid; //Turn the previous output layer into a sigmoid layer.
  n->output = newLayer;
  n->output->squish = softmax; //Set the new output layer to a softmax layer.
  if(n->input == NULL) n->input = newLayer;

}

* 
 * Description: Sets the inputs of the network.
 * n: the pointer to the network.
 * arr: the array of values to be passed into the network.
 *
void setInputs(RNN *n, char* arr){
  set_outputs(n->input, arr);
}
* 
 * Description: Does feed-forward, cost, and then backpropagation (gradient descent)
 * n: the pointer to the RNN.
 * label: the neuron expected to fire (for example, 4 for neuron 4)
 *
float descend(RNN *n, int label){
  feedforward(n);
	n->recurrent = n->output;
  return backpropagate_through_time(n->output, label, n->plasticity);
}
*/

static void set_recurrent_inputs(RNN *n){
	Layer *current = (Layer*)n->input->output_layer;
  int	recurrentInputs = 0;
	
	while(current != NULL){
		recurrentInputs += current->size;
		current = (Layer*)current->output_layer;
	}
	
	current = (Layer*)n->input->output_layer;
	int idx = n->input->size - recurrentInputs;
	int count = 0;
	for(int i = idx; i < n->input->size; i++){
		if(count > current->size){
			count = 0;
			current = (Layer*)current->output_layer;
		}
		n->input->neurons[i].activation = current->neurons[count].activation;
		count++;
	}
}

float step(RNN *n, int label){
	feedforward(n);
	set_recurrent_inputs(n);

}

/*
 *
 * Description: Performs the feed-forward operation on the network.
 * NOTE: setInputs should be used before calling feedforward.
 * n: A pointer to the network.
 *
void feedforward(RNN *n){
  Layer *current = n->input->output_layer;
  while(current != NULL){
    calculate_outputs(current);
    current = current->output_layer;
  }
}
*/
