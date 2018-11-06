/* Author: Jonah Siekmann
 * This is an example of how you might use a genetic algorithm rather than backpropagation
 * to train a neural network.
 * For simplicity's sake, we'll be using the same problem as in binary.c
 */

#include "MLP.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define POOL_SIZE 50
#define INPUT_DIMENSIONS 4
#define OUTPUT_DIMENSIONS 16
#define TRIALS 10
MLP pool[POOL_SIZE];

float fit_calc(MLP *n, int label){
	float sum = 0;
	for(int i = 0; i < n->output->size; i++){
		if(i == label){
			sum += n->output->neurons[i].activation;
		}
		else{
			sum -= n->output->neurons[i].activation;
		}
	}
	return sum;
}

Neuron *neuron_lookup(MLP *n, int layer_idx, int neuron_idx){
	printf("  starting neuron lookup, %d, %d\n", layer_idx, neuron_idx);
	Layer *current = n->input;
	for(int i = 0; i < layer_idx; i++){
		printf("	cycling thru %p\n", current->output_layer);
		current = current->output_layer;
	}
	printf("  returning neuron %d (size %lu)\n", neuron_idx, current->size);
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
	printf("starting crossbreed...\n");
	MLP ret = copy_mlp(partner1);
	printf("copied partner1\n");

	Layer *current = ret.input;
	int layer_idx = 0;
	while(current != NULL){
		printf("doing layer %d...\n", layer_idx);
		if(current->input_layer != NULL){
			printf("	Considering %p\n", current);
			for(int i = 0; i < current->size; i++){
				Neuron *partner_neuron = neuron_lookup(partner2, layer_idx, i);
				printf("got neuron %p (%d of %d), input layer %p\n", partner_neuron, i, current->size, current->input_layer);
				for(int j = 0; j < current->input_layer->size; j++){
					printf("considering weight %d\n", j);
					printf("	%p, %p, ", ret.output, ret.output->neurons);
					printf("	%p\n", partner_neuron);
					if(0) ret.output->neurons[i].weights[j] = partner_neuron->weights[j];
					printf("done with weight %d\n", j);
				}
				printf("done with loop\n");
				ret.output->neurons[i].bias = current->neurons[i].bias;
				if(0) ret.output->neurons[i].bias = partner_neuron->bias;
			}
		}
		layer_idx++;
		current = current->output_layer;
		printf("moving on to %p\n", current);
	}
	return ret;
}

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
	if(f1.performance < f2.performance) return 1;
	if(f1.performance > f2.performance) return -1;
	else return 0;
}
float sumweights(Layer* layer){
	float sum = 0;
	for(int i = 0; i < layer->size; i++){
		for(int j = 0; j < layer->input_layer->size; j++){
			sum += layer->neurons[i].weights[j];
		}
	}
	return sum;
}
void print_pool(MLP *pool, size_t len){
	printf("pool: [");
	for(int i = 0; i < len; i++){
		printf("%7.5f", sumweights(pool[i].output));
		if(i < len-1) printf(", ");
	}
	printf("]\n");
}
int main(void){	
	srand(time(NULL));
	// Create a pool of NN's with randomly initialized weights
	for(int i = 0; i < POOL_SIZE; i++){
		pool[i] = createMLP(INPUT_DIMENSIONS, 30, OUTPUT_DIMENSIONS);
	}
	float avg_fitness = 0;
	float last_fitness = 0;
	for(int i = 0; /* forever */; i++){ 
		for(int j = 0; j < POOL_SIZE; j++){
			MLP *n = &pool[j];
			n->performance = 0;
			for(int k = 0; k < TRIALS; k++){
				//Create a random 4-bit binary number
				int bit0 = rand()%2==0;
				int bit1 = rand()%2==0;
				int bit2 = rand()%2==0;
				int bit3 = rand()%2==0;
				int ans = bit0 * pow(2, 0) + bit1 * pow(2, 1) + bit2 * pow(2, 2) + bit3 * pow(2, 3);
				float arr[4] = {bit0, bit1, bit2, bit3}; //Input array (1 bit per input)

				setInputs(n, arr);
				feedforward(n); //Calculate outputs
				n->performance += fit_calc(n, ans)/TRIALS;
			}
		}
		qsort(pool, POOL_SIZE, sizeof(MLP), comp); //Order pool by highest fitness
		for(int j = 0; j < POOL_SIZE && (!(i%10000)); j++){
			printf("%d: %f\n", j, pool[j].performance);
		}
//		printf("Pool before cull:\n");
//		print_pool(pool, POOL_SIZE);
		for(int j = POOL_SIZE/2; j < POOL_SIZE; j++){
			printf("Killing %f due to subpar performance (%f)\n", sumweights(pool[j].output), pool[j].performance);
			dealloc_network(&pool[j]);
			printf("Dealloc done. now choosing parents\n");
			MLP *parent1 = &pool[rand()%(POOL_SIZE/2)];
			MLP *parent2 = &pool[rand()%(POOL_SIZE/2)];
			printf("crossbreeding...\n");
			pool[j] = crossbreed(parent1, parent2);
			printf("crossbred. now mutating...\n");
			mutate(pool[j].output, 0.05, 0.01);
			printf("done\n");
		}
//		printf("Pool after cull:\n");
//		print_pool(pool, POOL_SIZE);
		float fitness = pool[0].performance;
		avg_fitness += fitness;
//		float improvement = 100 * ((last_fitness/fitness) - 1);
		//Debug stuff
		if(!(i % 1)){
			float similarity = similarity_score(&pool[0], &pool[POOL_SIZE-1]);
			printf("CURRENTLY ON GENERATION %d, best fitness: %f, avg %f, similarity between best and worst network: %f\n", i, pool[0].performance, avg_fitness/i, similarity);
			getchar();
		}	
	}
}
