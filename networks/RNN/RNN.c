
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
	for(int i = 0; i < size-1; i++){
		input_size += arr[i];
	}
	for(int i = 0; i < size; i++){
		if(i == 0) addLayer(&n, input_size);
		else addLayer(&n, arr[i]);
	}
  return n;
}


static size_t recurrentInputSize(RNN *n){
	Layer *current = (Layer*)n->input->output_layer;
  int	sum = 0;
	
	while(current != n->output && current != NULL){
		sum += current->size;
		current = (Layer*)current->output_layer;
	}
	return sum;
}
void setOneHotInput(RNN *n, float *arr){
	size_t recurrentInputIndex = n->input->size - recurrentInputSize(n);
	for(int i = 0; i < recurrentInputIndex; i++){
		n->input->neurons[i].activation = arr[i];
	}
}

static void set_recurrent_inputs(RNN *n){
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
}

float step(RNN *n, int label){
	feedforward_recurrent(n);
	return backpropagate(n->output, label, n->plasticity);
}

/*
 *
 * Description: Performs the feed-forward operation on the network.
 * NOTE: setInputs should be used before calling feedforward.
 * n: A pointer to the network.
 */
void feedforward_recurrent(RNN *n){
	set_recurrent_inputs(n);
	feedforward(n);
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
