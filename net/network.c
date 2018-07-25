/* Author: Jonah Siekmann
 * 7/24/2018
 * This is a basic feed-forward neural network implementation using backpropagation. I've tested it with mnist and a few trivial problems.
 * Every function beginning with static is meant for internal use only. You may call any other function.
 */

#include "network.h"
#include <math.h>


/* Description: Allocates an array of Neurons, and then allocates and initializes each Neuron's weights and biases.
* size: the size of the layer currently being build
* previousLayer: the preceding layer in the network (can be NULL).
*
*/
static Layer *createLayer(size_t size, Layer *previousLayer){
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
      weights[j] = ((float)(rand()%7)-3.2); //Slight bias towards positive weights
    }
    neurons[i].weights = weights;
    neurons[i].bias = ((float)(rand()%7)-3.5);
  }
  layer->size = size;
  layer->neurons = neurons;
  layer->input_layer = previousLayer;
  return layer;
}
static float squish(float input){
  return 1 / (1 + exp(-input));
}

/* Description: Sets the outputs of a layer manually, instead of calculating them.
*              The purpose of this function is to set the outputs of the input layer.
* layer: The layer whose outputs will be set.
* outputs: The array of outputs which will be assigned to the layer
* important: dimensions of outputs and layer must both be size*size
*/
static void setOutputs(Layer *layer, float *outputs){
  for(int i = 0; i < layer->size; i++){
    layer->neurons[i].output = outputs[i];//squish(outputs[i]);
    layer->neurons[i].dOutput = squish(layer->neurons[i].input) * (1 - squish(layer->neurons[i].input));
  }
}

/* Description: Performs backpropagation algorithm on the network.
* output_layer: The last layer in the network. You must call cost() on this layer before calling backpropagate.
*
*/
static void backpropagate(Layer *output_layer, float plasticity){
  Layer *current = output_layer;
  while(current->input_layer != NULL){
    Layer* input_layer = (Layer*)(current->input_layer);
    for(int i = 0; i < input_layer->size; i++){
      //Calculate activation gradients in input layer BEFORE doing nudges to weights and biases in the current layer
      float sum = 0;
      for(int j = 0; j < current->size; j++){
        float weight = current->neurons[j].weights[i];
        float dSig = current->neurons[j].dOutput;
        float activationGradient = current->neurons[j].activationGradient;
        sum += weight*dSig*activationGradient;
      }
      input_layer->neurons[i].activationGradient = sum*plasticity;
    }

    for(int i = 0; i < current->size; i++){
      Neuron *currentNeuron = &current->neurons[i];
      float dSig = currentNeuron->dOutput;
      float activationGradient = currentNeuron->activationGradient;

      //Calculate weight nudges
      for(int j = 0; j < input_layer->size; j++){
        float a = input_layer->neurons[j].output;
        float in = input_layer->neurons[j].input;
        currentNeuron->weights[j] += a*dSig*activationGradient*plasticity;
      }
      //Calculate bias nudge
      currentNeuron->bias += dSig * activationGradient*plasticity;
    }
    current = current->input_layer;
  }
}


/* Description: Calculates the outputs of each neuron in a layer based on the outputs & weights of the neurons in the preceding layer
* layer1: the layer with existing outputs
* layer2: the layer whose outputs will be calculated
* important: dimensions of weights of layer2 must match dimensions of neurons of layer1
*/
static void calculateOutputs(Layer *layer){
  Layer *input_layer = (Layer*)(layer->input_layer);
  for(int i = 0; i < layer->size; i++){
    float sum = 0;
    Neuron *current = &(layer->neurons[i]);
    for(int k = 0; k < input_layer->size; k++){
      sum += input_layer->neurons[k].output * layer->neurons[i].weights[k];
    }
    current->input = sum + current->bias;
    current->dOutput = squish(current->input) * (1 - squish(current->input));
    current->output = squish(current->input);
  }
}

/* Description: Calculates the cost of the output layer and sets the activation gradients for the output neurons.
* output_layer: the last layer in the network.
* label: the expected value chosen by the network.
*/
static float cost(Layer *output_layer, int label){
  float sum = 0;
  for(int i = 0; i < output_layer->size; i++){
    int y = (i==label);
    sum += (output_layer->neurons[i].output - y)*(output_layer->neurons[i].output - y);
    output_layer->neurons[i].activationGradient = -2 * (output_layer->neurons[i].output - y);
  }
  return sum;
}
/* Description: Initializes a network object.
*
*/
Network initNetwork(){
  Network n;
  n.input = NULL;
  n.output = NULL;
  return n;
}

/* Description: Adds a layer to the network object. Layers will be inserted from input layer onward.
*  n: the pointer to the network.
*  size: the desired size of the layer to be added.
*/

void addLayer(Network *n, size_t size){
  Layer *newLayer = createLayer(size, n->output);
  n->output = newLayer;
  if(n->input == NULL) n->input = newLayer;

}

/* Description: Sets the inputs of the network.
*  n: the pointer to the network.
*  arr: the array of values to be passed into the network.
*/
void setInputs(Network *n, float* arr){
  setOutputs(n->input, arr);
}

/* Description: Does feed-forward, cost, and then backpropagation.
*  n: the pointer to the network.
*  label: the neuron expected to fire (4 for neuron 4)
*/
float runEpoch(Network *n, int label){
  Layer *current = n->input->output_layer;
  while(current != NULL){
    calculateOutputs(current);
    current = current->output_layer;
  }
  float c = cost(n->output, label);
  backpropagate(n->output, 1);
  return c;
}

void printWeights(Layer *layer){
  printf("weights:\n");
  Layer *previousLayer = (Layer*)layer->input_layer;
  for(int i = 0; i < previousLayer->size; i++){
    printf("  Weight for neuron %d with output %f is %f\n", i, previousLayer->neurons[i].output, layer->neurons[1].weights[i]);
  }
}

void printActivationGradients(Layer *layer){
  printf("activation gradients:\n");
  for(int i = 0; i < layer->size; i++){
    printf("  Neuron %d: %f from %f\n", i, layer->neurons[i].activationGradient, layer->neurons[i].output);
  }
}
void printOutputs(Layer *layer){
  for(int i = 0; i < layer->size; i++){
    float val = layer->neurons[i].output;
      printf("Neuron %d output: %f, bias: %f\n", i, val, squish(layer->neurons[i].bias));
		}
		printf("\n");
}
void prettyprint(Layer *layer){
  for(int i = 0; i < layer->size; i++){
    float val = layer->neurons[i].output;
    if(!(i % (int)sqrt(layer->size))) printf("\n");
		if(val <= 0.5) printf(".");
    if(val > 0.5) printf("A");
		}
    printf("\n");
}
