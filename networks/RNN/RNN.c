
/* Author: Jonah Siekmann
 * 8/10/2018
 * This is an attempt at writing a recurrent neural network (RNN) Every function beginning with static is meant for internal use only. You may call any other function.
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


static size_t recurrentInputSize(Layer* layer){
  Layer* current = layer;
//	printf("		recurrentInputSize: Layer being considered: %p, input layer: %p\n", layer, layer->input_layer);
//	printf("		recurrentInputSize: Layers: %p --> %p --> %p\n", layer, layer->output_layer, ((Layer*)layer->output_layer)->output_layer);
	while(((Layer*)current->output_layer)->output_layer != NULL) current = current->output_layer;
//	printf("		recurrentInputSize: Layer selected to work back from %p\n", current);
	
	size_t recurrent_offset;
	size_t temp = 0;
	while(current != layer->input_layer){
		recurrent_offset = current->size - temp;
		temp = recurrent_offset;
//		printf("		recurrentInputSize: On layer %p, recurrent offset is %lu\n", current, recurrent_offset);
		current = current->input_layer;
	}
//	printf("		recurrentInputSize: returning %lu\n", recurrent_offset);
	return recurrent_offset;
	/*
  while(((Layer*)current->output_layer)->output_layer != NULL) current = current->output_layer;

	size_t recurrent_neurons_starting_index;
	size_t temp = 0;
	while(current != layer->input_layer){
		recurrent_neurons_starting_index = current->size - temp; 
		temp = recurrent_neurons_starting_index; 
		current = current->input_layer;
	}
	printf("recurrent offset: %lu\n", recurrent_neurons_starting_index);
	return recurrent_neurons_starting_index;
*/
/*
	Layer *current = (Layer*)n->input->output_layer;
	int sum = 0;
	
	while(current != n->output && current != NULL){
		sum += current->size;
		current = (Layer*)current->output_layer;
	}
	return sum;
*/
}
void setOneHotInput(RNN *n, float *arr){
	size_t recurrentInputIndex = recurrentInputSize(n->input);
//	printf("Setting recurrent inputs of layer %p, starting at index %lu\n", n->input, recurrentInputIndex);
	for(int i = 0; i < recurrentInputIndex; i++){
		n->input->neurons[i].activation = arr[i];
	}
}

static void set_recurrent_inputs(Layer* layer){
//	printf("	set_recurrent_inputs: Setting recurrent inputs for layer %p, input: %p\n", layer, layer->input_layer);
	size_t recurrent_offset = recurrentInputSize(layer->input_layer);
	if(layer->input_layer != NULL){
//		printf("	set_recurrent_inputs: using offset of %lu for layer %p\n", recurrent_offset, layer->input_layer);

		Layer *input_layer = (Layer*)layer->input_layer;
		for(int i = 0; i < input_layer->size; i++){
//			printf("	set_recurrent_inputs: before assignment neuron %d with val %f in layer %p\n", i, input_layer->neurons[i].activation, input_layer);
		}
		for(int i = recurrent_offset; i < input_layer->size; i++){
			Neuron *recurrent_neuron = &input_layer->neurons[i];
			Neuron *old_neuron = &layer->neurons[i-recurrent_offset];
			input_layer->neurons[i].activation = layer->neurons[i-recurrent_offset].activation;
//			printf("	set_recurrent_inputs: neuron %d of layer %p (%f) set to val of neuron %d of layer %p (%f)\n", i, input_layer, recurrent_neuron->activation, i-recurrent_offset, layer, old_neuron->activation);
			//printf("WE GOT A FUCKIN PROBLEM BOIS: %f\n", input_layer->neurons[i].activation);
			//getchar();
		}
//		for(int i = 0; i < input_layer->size; i++){
//			printf("	set_recurrent_inputs: AFTER ASSIGNMENT NEURON %d HAS VAL %f\n", i, input_layer->neurons[i].activation);
//		}
	}
	/*
	size_t recurrentInputs = recurrentInputSize(n);

	Layer *current = (Layer*)n->input->output_layer;
	int idx = n->input->size - recurrentInputs;
	int count = 0;
	for(int i = idx; i < n->input->size; i++){
		if(count > current->size){
			count = 0;
			current = (Layer*)current->output_layer;
		}
		n->input->neurons[i].activation = current->neurons[count].activation;
//		printf("Setting input neuron %d to hidden neuron %d,  %f -> %f\n", i, count, current->neurons[count].activation, n->input->neurons[i].activation);
		count++;
	}
  */
}

float step(RNN *n, int label){
	for(int i = 0; i < n->input->size; i++){
		Layer *layer = n->input;
//		printf("Neuron %d of layer %p has output %f\n", i, layer, layer->neurons[i].activation);
	}	
	feedforward_recurrent(n);
	return backpropagate(n->output, label, n->plasticity);
}

/*
 *
 * Description: Performs the feed-forward operation on the network and sets
 *              recurrent inputs from previous hidden state of layers.
 * NOTE: setInputs should be used before calling feedforward_recurrent.
 * n: A pointer to the network.
 */
void feedforward_recurrent(RNN *n){
//	set_recurrent_inputs(n->input);

	Layer* current = n->input->output_layer;
	while(current != NULL){
//		printf("feedforward_recurrent: setting recurrent inputs for %p\n", current);
		/*if(current != n->output/* && current != ((Layer*)n->output)->input_layer*/ set_recurrent_inputs(current);
		calculate_outputs(current);
		current = current->output_layer;
//		printf("feedforward_recurrent: done with %p\n", current);
	}
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
