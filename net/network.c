#include "network.h"
#include <math.h>


/* Description: Allocates an array of Neurons, and then allocates and initializes each Neuron's weights and biases.
 * size: the size of the layer currently being build
 * previous_layer_size: the size of the preceding layer in the network (can be 0).
 *
 */
Layer *create_layer(size_t size, Layer *previousLayer){
  Layer *layer = (Layer*)malloc(size*sizeof(Layer));
  size_t previous_layer_size = 0;
  if(previousLayer != NULL) previous_layer_size = previousLayer->size;

  //Allocate every neuron in the layer
  Neuron* neurons = (Neuron*)malloc(size*sizeof(Neuron));
  for(int i = 0; i < size; i++){
    //Allocate weights
    float *weights = (float*)malloc(previous_layer_size*sizeof(float));
    for(int j = 0; j < previous_layer_size; j++){
        weights[j] = ((float)(rand()%700)-350)/100;
    }
    neurons[i].weights = weights;
    neurons[i].bias = ((float)(rand()%700)-350)/100;
  }
  layer->size = size;
  layer->neurons = neurons;
  layer->input_layer = previousLayer;
  return layer;
}

/* Description: Sets the outputs of a layer manually, instead of calculating them.
 *              The purpose of this function is to set the outputs of the input layer.
 * layer: The layer whose outputs will be set.
 * outputs: The array of outputs which will be assigned to the layer
 * important: dimensions of outputs and layer must both be size*size
 */
void setOutputs(Layer *layer, float *outputs){
  for(int i = 0; i < layer->size; i++){
    layer->neurons[i].output = outputs[i];
	}
}

/* Description: Performs backpropagation algorithm on the network. cost() must be called before performing this.
 *
 *
 */
void backpropagate(Layer *output_layer){
  Layer *current = output_layer;
  while(current->input_layer != NULL){
    Layer* input_layer = (Layer*)(current->input_layer);
    for(int i = 0; i < input_layer->size; i++){
      //Calculate activation gradients in input layer BEFORE doing nudges to weights in the current layer
      float sum = 0;
      for(int j = 0; j < current->size; j++){
        float weight = current->neurons[j].weights[i];
        float dSig = current->neurons[j].dOutput;
        float activationGradient = current->neurons[j].activationGradient;
        sum += weight*dSig*activationGradient;
      }
      input_layer->neurons[i].activationGradient = sum;
    }
    
    for(int i = 0; i < current->size; i++){
      Neuron *currentNeuron = &current->neurons[i];
      float dSig = currentNeuron->dOutput;
      float activationGradient = currentNeuron->activationGradient;

      //Calculate bias nudge
      currentNeuron->bias += 1 * dSig * activationGradient;

      //Calculate weight nudges
      for(int j = 0; j < input_layer->size; j++){
        float a = input_layer->neurons[j].output;
        currentNeuron->weights[j] += a*dSig*activationGradient;
      }
    }
    current = current->input_layer;
  }
}

float squish(float input){
	return 1 / (1 + exp(input));
}

float dsqish(float input){
  float squished = squish(input);
  return squished * (1 - squished);
}
/* Description: Calculates the outputs of each neuron in a layer based on the outputs & weights of the neurons in the preceding layer
 * layer1: the layer with existing outputs
 * layer2: the layer whose outputs will be calculated
 * important: dimensions of weights of layer2 must match dimensions of neurons of layer1
 */
void calculateOutputs(Layer *layer){
  Layer *input_layer = (Layer*)(layer->input_layer);
  for(int i = 0; i < layer->size; i++){
		float sum = 0;
    Neuron *current = &(layer->neurons[i]);
		for(int k = 0; k < input_layer->size; k++){
			sum += input_layer->neurons[k].output * layer->neurons[i].weights[k];
		}
		current->output = squish(sum);
    current->dOutput = current->output * (1 - current->output);
	}
}

/* Description: Calculates the cost of the output layer and sets the activation gradients for the output neurons.
 * output_layer: the last layer in the network.
 * label: the expected value chosen by the network.
 */
float cost(Layer *output_layer, int label){
  float sum = 0;
  for(int i = 0; i < output_layer->size; i++){
    if(i!=label){
      sum += (output_layer->neurons[i].output*output_layer->neurons[i].output);
      output_layer->neurons[i].activationGradient = 2 * (output_layer->neurons[i].output - 0);
    }else{
      sum += (output_layer->neurons[i].output-1)*(output_layer->neurons[i].output-1);
      output_layer->neurons[i].activationGradient = 2 * (output_layer->neurons[i].output - 1);
    }
  }
  return sum;
}

void printOutputs(Layer *layer){
  for(int i = 0; i < layer->size; i++){
    float val = layer->neurons[i].output;
      if(layer->size > 10){
        if(!(i % (int)sqrt(layer->size))) printf("\n");
  			if(val < 10) printf(".");
        if(val > 10) printf("A");
      }else{
        printf("Neuron %d output: %f\n", i, val);
      }
		}
		printf("\n");
}
