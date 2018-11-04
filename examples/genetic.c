/* Author: Jonah Siekmann
 * This is an example of how you might use a genetic algorithm rather than backpropagation
 * to train a neural network.
 * For simplicity's sake, we'll be using the same problem as in binary.c
 */

#include "MLP.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define POOL_SIZE 10
#define INPUT_DIMENSIONS 4
#define OUTPUT_DIMENSIONS 16

MLP pool[10];

float fit_calc(MLP *n, int label){
	float sum = 0;
	for(int i = 0; i < n->output->size; i++){
		if(i == label) sum += 1 - n->output->neurons[i].activation;
		else sum += n->output->neurons[i].activation;
	}
	return sum;
}

MLP copy_mlp(MLP *n){
	MLP ret = initMLP();
	Layer *current = n->input;
	while(current != NULL){
		printf("Considering %p\n", current);
		addLayer(&ret, current->size);
		printf("adding squish at %p\n", ret.output->squish);
		ret.output->squish = current->squish;
		for(int i = 0; i < current->size; i++){
			printf("Considering neuron %p and %p\n", ret.output->neurons, current->neurons);
			ret.output->neurons[i].activation = current->neurons[i].activation;
			ret.output->neurons[i].dActivation = current->neurons[i].dActivation;
			ret.output->neurons[i].gradient = current->neurons[i].gradient;
			ret.output->neurons[i].input = current->neurons[i].input;

			if(current->input_layer != NULL){
				for(int j = 0; j < current->input_layer->size; j++){
					ret.output->neurons[i].weights[j] = current->neurons[i].weights[j];
				}
				ret.output->neurons[i].bias = current->neurons[i].bias;
			}
		}
		current = current->output_layer;
	}
	return ret;
}

int highest(float *arr, int len){
	int highestone = 0;
	for(int i = 0; i < len; i++){
		if(arr[i] < arr[highestone]) highestone = i;
	}
	return highestone;
}
int lowest(float *arr, int len){
	int lowestone = 0;
	for(int i = 0; i < len; i++){
		if(arr[i] < arr[lowestone]) lowestone = i;
	}
	return lowestone;
}

int main(void){	
	srand(time(NULL));

	// Create a pool of NN's with randomly initialized weights
	for(int i = 0; i < POOL_SIZE; i++){
		pool[i] = createMLP(INPUT_DIMENSIONS, 20, 20, OUTPUT_DIMENSIONS);
	}
	
	for(int i = 0; i < 80000000; i++){ //Run the network for 80000....00 examples
		//Create a random 4-bit binary number
		int bit0 = rand()%2==0;
		int bit1 = rand()%2==0;
		int bit2 = rand()%2==0;
		int bit3 = rand()%2==0;
		int ans = bit0 * pow(2, 0) + bit1 * pow(2, 1) + bit2 * pow(2, 2) + bit3 * pow(2, 3);

		float arr[4] = {bit0, bit1, bit2, bit3}; //Input array (1 bit per input)

		float fitnesses[POOL_SIZE];
		for(int j = 0; j < POOL_SIZE; j++){
			MLP *n = &pool[i];
			setInputs(n, arr);
			feedforward(n); //Calculate outputs
			gradients_wrt_outputs(n->output); //Gradients
			fitnesses[j] = fit_calc(n, ans);
			
		}
		int lowest_fitness_idx = lowest(fitnesses, POOL_SIZE);
		MLP *lowest_fitness = &pool[lowest_fitness_idx];
		dealloc_network(lowest_fitness);
		pool[lowest_fitness_idx] = copy_mlp(&pool[highest(fitnesses, POOL_SIZE)]);
		mutate(pool[lowest_fitness_idx].output, 0.01, 0.1);

		
		//Debug stuff
		if(!(i % 1000)){
			printf("CURRENTLY ON GENERATION %d, best fitness: %f\n", i, fitnesses[highest(fitnesses, POOL_SIZE)]);
			getchar();
		}	
	}
}
