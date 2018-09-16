
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
	set_recurrent_inputs(n);
	feedforward(n);
	return backpropagate(n->output, label, n->plasticity);
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
