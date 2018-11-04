/* Author: Jonah Siekmann
 * This is an example of how you might use a genetic algorithm rather than backpropagation
 * to train a neural network.
 * For simplicity's sake, we'll be using the same problem as in binary.c
 */

#include "MLP.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define POOL_SIZE 30
#define INPUT_DIMENSIONS 4
#define OUTPUT_DIMENSIONS 16

MLP pool[POOL_SIZE];

float fit_calc(MLP *n, int label){
	float sum = 0;
	for(int i = 0; i < n->output->size; i++){
		if(i == label) sum += 1 - n->output->neurons[i].activation;
		else sum += n->output->neurons[i].activation;
	}
	return sum;
}

Neuron *neuron_lookup(MLP *n, int layer_idx, int neuron_idx){
//	printf("  starting neuron lookup, %d, %d\n", layer_idx, neuron_idx);
	Layer *current = n->input;
	for(int i = 0; i < layer_idx; i++){
//		printf("	cycling thru %p\n", current->output_layer);
		current = current->output_layer;
	}
//	printf("  returning neuron %d (size %lu)\n", neuron_idx, current->size);
	return &current->neurons[neuron_idx];
}

MLP copy_mlp(MLP *n){
	MLP ret = initMLP();
	Layer *current = n->input;

	while(current != NULL){
		addLayer(&ret, current->size);
		ret.output->squish = current->squish;
		for(int i = 0; i < current->size; i++){
			/*
			ret.output->neurons[i].activation = current->neurons[i].activation;
			ret.output->neurons[i].dActivation = current->neurons[i].dActivation;
			ret.output->neurons[i].gradient = current->neurons[i].gradient;
			ret.output->neurons[i].input = current->neurons[i].input;
			*/

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

void print_layers(MLP *n){
	printf("Layers & sizes for network %p: [", n);
	Layer *pr = n->input;
	while(pr != NULL){
		printf("%p (%lu)", pr, pr->size);
		pr = pr->output_layer;
		if(pr) printf(", ");
	}
	printf("]\n");
}
MLP crossbreed(MLP *partner1, MLP *partner2){
	MLP ret = copy_mlp(partner1);

	Layer *current = ret.input;
	int layer_idx = 0;
	while(layer_idx < 4){
		ret.output->squish = current->squish;
		if(current->input_layer != NULL){

			for(int i = 0; i < current->size; i++){
				Neuron *partner_neuron = neuron_lookup(partner2, layer_idx, i);
				for(int j = 0; j < current->input_layer->size; j++){
					if(rand()&1) ret.output->neurons[i].weights[j] = partner_neuron->weights[j];
				}
				ret.output->neurons[i].bias = current->neurons[i].bias;
				if(rand()&1) ret.output->neurons[i].bias = partner_neuron->bias;
			}
		}
		layer_idx++;
		current = current->output_layer;
	}
	return ret;
}
/*
int highest(MLP *pool, int len){
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
*/
float similarity_score(MLP *a, MLP *b){
	int total = 0;
	int identical = 0;
	int layeridx = 0;
	Layer *current = a->input;
	while(current != NULL){
		if(current->input_layer != NULL){
			for(int i = 0; i < current->size; i++){
				Neuron *comparator = neuron_lookup(b, layeridx, i);
				for(int j = 0; j < current->input_layer->size; j++){
					if(comparator->weights[j] == current->neurons[i].weights[j]){
							identical++;
					}
					total++;
				}
			}	
		}
		current = current->output_layer;
		layeridx++;
	}
	return ((float)identical)/total;
}
int comp(const void *mlp1, const void *mlp2){
	MLP f1 = *(MLP*)mlp1;
	MLP f2 = *(MLP*)mlp2;
	if(f1.performance > f2.performance) return 1;
	if(f1.performance < f2.performance) return -1;
	else return 0;
}

int main(void){	
	srand(time(NULL));
	// Create a pool of NN's with randomly initialized weights
	for(int i = 0; i < POOL_SIZE; i++){
		pool[i] = createMLP(INPUT_DIMENSIONS, 15, 15, OUTPUT_DIMENSIONS);
	}
	float avg_fitness = 0;
	float last_fitness = 0;
	for(int i = 0; /* forever */; i++){ 
		//Create a random 4-bit binary number
		int bit0 = rand()%2==0;
		int bit1 = rand()%2==0;
		int bit2 = rand()%2==0;
		int bit3 = rand()%2==0;
		int ans = bit0 * pow(2, 0) + bit1 * pow(2, 1) + bit2 * pow(2, 2) + bit3 * pow(2, 3);

		float arr[4] = {bit0, bit1, bit2, bit3}; //Input array (1 bit per input)

		for(int j = 0; j < POOL_SIZE; j++){
			MLP *n = &pool[j];
			setInputs(n, arr);
			feedforward(n); //Calculate outputs
			n->performance = fit_calc(n, ans);
		}
		qsort(pool, POOL_SIZE, sizeof(MLP), comp); //Order pool by highest fitness
		for(int j = 0; j < POOL_SIZE && (!(i%10000)); j++){
			printf("%d: %f\n", j, pool[j].performance);
		}
		for(int j = POOL_SIZE/2; j < POOL_SIZE; j++){
			dealloc_network(&pool[j]);
			MLP *parent1 = &pool[rand()%(POOL_SIZE/2)];
			MLP *parent2 = &pool[rand()%(POOL_SIZE/2)];
			pool[j] = crossbreed(parent1, parent2);
			mutate(pool[j].output, 0.05, 0.001);
		}
		float fitness = pool[0].performance;
		avg_fitness += fitness;
//		float improvement = 100 * ((last_fitness/fitness) - 1);
		//Debug stuff
		if(!(i % 10000)){
			printf("CURRENTLY ON GENERATION %d, best fitness: %f, avg %f\n", i, pool[0].performance, avg_fitness/i);
			getchar();
		}	
	}
}
