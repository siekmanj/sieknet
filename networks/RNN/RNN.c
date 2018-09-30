
/* Author: Jonah Siekmann
 * 8/10/2018
 * This is an attempt at writing a recurrent neural network (RNN) Every function beginning with static is meant for internal use only. You may call any other function.
 *
 * I am fairly certain this is not a 'full' RNN, since the network only has access to the previous timestep's state, and thus only needs to do one feedforward and one
 * backprop operation per timestep - most descriptions of RNN's seem to mention multiple feedforward/backprops per timestep. It is possible this implementation is
 * more similar to an Elman network than a true RNN. If anyone can clarify or offer insight, I would be very grateful.
 * 
 * Some confusion may arise from the fact that many of the functions used in this implementation are defined in MLP.c, like backpropagate. Instead of rewriting these
 * functions, I have elected to build on the basic multilayer perceptron and re-use these functions.
 * As of 9/20/2018, this appears to be working in a somewhat stable fashion. If you get nans at any point, consider changing your n.plasticity. 
 */

#include "RNN.h"
#include <math.h>
#include <string.h>

/* 
 * Description: Initializes a recurrent neural network object.
 */
static RNN initRNN(){
  RNN n;
  n.input = NULL;
  n.output = NULL;
  n.performance = 0;
  n.plasticity = .05;
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
	for(int i = 0; i < size-2; i++){
		addLayer(&n, arr[i] + arr[i+1]); //Add a layer but allocate extra neurons for the hidden state of next layer. 
	}	
	addLayer(&n, arr[size-2]); //Add the layer right before the output layer which is not used as a recurrent input for the output layer.
	for(int i = 0; i < n.input->size; i++){
		Layer *layer = n.input;
	}	
	addLayer(&n, arr[size-1]); //Add an output layer that is not used as a recurrent input.
  return n;
}


/*
 * Description: Calculates the offset at which the hidden state of the next layer starts. 
 * layer: A pointer to the layer for which the recurrent input offset will be calculated.
 */
static size_t recurrent_input_offset(Layer* layer){
  Layer* current = layer;
	while(((Layer*)current->output_layer)->output_layer != NULL) current = current->output_layer;
	
	size_t recurrent_offset;
	size_t temp = 0;
	while(current != layer->input_layer){
		recurrent_offset = current->size - temp;
		temp = recurrent_offset;
		current = current->input_layer;
	}
	return recurrent_offset;
}

/*
 * Description: Sets the activations of the neurons in the input layer of the network.
 * n: The pointer to the rnn.
 * arr: An array of floats (of which all but one should be 0.0, and the other 1.0, depending on your use case)
 * NOTE: setInputs in MLP.c works similarly, but doesn't take into account the fact that the input layer
 *       includes the hidden state of its output layer - therefore the size of the layer will be larger than
 *       the array size, possibly leading to weird behavior.
 */
void setOneHotInput(RNN *n, float *arr){
	size_t recurrentInputIndex = recurrent_input_offset(n->input);

	for(int i = 0; i < recurrentInputIndex; i++){
		n->input->neurons[i].activation = arr[i];
	}
}

/*
 * Description: Sets the recurrent inputs in the input layer of the provided layer.
 * layer: The pointer to the layer for which recurrent inputs will be set (layer->input_layer should not be null)
 */
static void set_recurrent_inputs(Layer* layer){
	size_t recurrent_offset = recurrent_input_offset(layer->input_layer);
	if(layer->input_layer != NULL){

		Layer *input_layer = (Layer*)layer->input_layer;
		for(int i = recurrent_offset; i < input_layer->size; i++){
			Neuron *recurrent_neuron = &input_layer->neurons[i];
			Neuron *old_neuron = &layer->neurons[i-recurrent_offset];

			recurrent_neuron->activation = old_neuron->activation;
		}
	}
}

/* 
 * Description: This is the RNN equivalent of descend in mlp.c. It performs a feedforward operation
 *              in which the recurrent inputs of recurrent layers are set, then does backpropagation.
 * n: the pointer to the rnn
 * label: the neuron expected to fire.
 * NOTE: setInputs should be used before calling step().
 */
float step(RNN *n, int label){
	feedforward_recurrent(n);
	return backpropagate(n->output, label, n->plasticity);
}

/*
 *
 * Description: Performs the feed-forward operation on the network and sets
 *              recurrent inputs from previous hidden state of layers.
 * n: A pointer to the network.
 * NOTE: setInputs should be used before calling feedforward_recurrent().
 */
void feedforward_recurrent(RNN *n){
	Layer* current = n->input->output_layer;
	while(current != NULL){
		set_recurrent_inputs(current);
		calculate_outputs(current);
		current = current->output_layer;
	}
}

/*
 * IO FUNCTIONS FOR READING AND WRITING TO A FILE
 * NOTE: Since these are copy+pasted from MLP.c, they will be replaced once I find a cleaner solution.
 */

static void writeToFile(FILE *fp, char *ptr){
  fprintf(fp, "%s", ptr);
  memset(ptr, '\0', strlen(ptr));
}

static void getWord(FILE *fp, char* dest){
  memset(dest, '\0', strlen(dest));
  //printf("bytes read: %lu\n", fread(dest, 1024, 1, fp));
  fscanf(fp, " %1023s", dest);
}
/* 
 * Description: Saves the network's state to a file that can be read later.
 * n: A pointer to the network.
 * filename: The desired filename and path.
 */
void saveRNNToFile(RNN *n, char* filename){
 FILE *fp;
 char buff[1024];
 memset(buff, '\0', 1024);

 //Create file
 fp = fopen(filename, "w");
 printf("Saving to: %s\n", filename);
 memset(buff, '\0', strlen(buff));

 //Get network dimensions
 size_t size = 0;
 Layer *current = n->input;
 while(1){
   if(current != NULL) size++;
   if(current == n->output) break;
   current = current->output_layer;
 }

 //Write header info to file
 strcat(buff, "RNN ");
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
       Layer* input_layer = (Layer*)current->input_layer;

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
 */
RNN loadRNNFromFile(const char *filename){
  FILE *fp = fopen(filename, "rb");
  char buff[1024];
  memset(buff, '\0', 1024);

  RNN n = initRNN();
  getWord(fp, buff); //Get first word to check if RNN file

  if(strcmp(buff, "RNN") != 0){
    printf("ERROR: [%s] is not RNN.\n", buff);
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
    addLayer(&n, layer_size);

    for(int j = 0; j < layer_size; j++){
      getWord(fp, buff);
      if(strcmp(buff, "INPUTLAYER") == 0) {
        break;
      }
      getWord(fp, buff);
      size_t number_of_weights = strtol(buff, NULL, 10);
      Layer *input_layer = (Layer*)n.output->input_layer;
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
