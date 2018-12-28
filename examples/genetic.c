/* Author: Jonah Siekmann
 * This is an example of how you might use a genetic algorithm rather than backpropagation
 * to train a neural network.
 * For simplicity's sake, we'll be using the same problem as in binary.c
 */

#include "MLP.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define POOL_SIZE 65
#define INPUT_DIMENSIONS 4
#define OUTPUT_DIMENSIONS 16
#define TRIALS 10

//Mutation rate of 0.09 and learning rate of 0.05 seem to work well.
#define MUTATION_RATE 0.1
#define LEARNING_RATE 0.25

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
	Layer *current = n->input;
	for(int i = 0; i < layer_idx; i++){
		current = current->output_layer;
	}
	return &current->neurons[neuron_idx];
}

void print_layers(MLP *n){
	printf("network %p: [", n);
	Layer *pr = n->input;
	while(pr != NULL){
		printf("%p (%lu)", pr, pr->size);
		pr = pr->output_layer;
		if(pr) printf(", ");
	}
	printf("]\n");
}

MLP copy_mlp(MLP *n){
	MLP ret = initMLP();
	Layer *current = n->input;

	while(current != NULL){
		addLayer(&ret, current->size);
		ret.output->squish = current->squish;
		if(current->input_layer != NULL){
			for(int i = 0; i < current->size; i++){
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

MLP crossbreed(MLP *partner1, MLP *partner2){
//	printf("copying...\n");
	MLP ret = copy_mlp(partner1);
//	printf("copy done,\n");
//	print_layers(&ret);
	Layer *current = ret.input;
	int layer_idx = 0;
	while(current != NULL){
//		printf("Starting layer %d, pointer: %p, inpt: %p\n", layer_idx, current, current->input_layer);
		if(current->input_layer != NULL){
			for(int i = 0; i < current->size; i++){
				Neuron *partner_neuron = neuron_lookup(partner2, layer_idx, i);
				for(int j = 0; j < current->input_layer->size; j++){
//					printf("	Doing layer %d, neuron %d, weight %d\n", layer_idx, i, j);
					if(!(rand()%2)) current->neurons[i].weights[j] = partner_neuron->weights[j];
				}
				if(!(rand()%2)) current->neurons[i].bias = partner_neuron->bias;
//				printf("	Done with neuron %d\n", i);
			}
		}
//		printf("	Done with layer %d\n", layer_idx);
		layer_idx++;
		current = current->output_layer;
//		printf("	Moving on to layer %d\n", layer_idx);
	}
//	printf("Exiting crossbreed\n");
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
	setbuf(stdout, NULL);
	/*
	MLP a = createMLP(INPUT_DIMENSIONS, 30, OUTPUT_DIMENSIONS);
	MLP b = createMLP(INPUT_DIMENSIONS, 30, OUTPUT_DIMENSIONS);
	MLP c = crossbreed(&a, &b);
	dealloc_network(&a);
	dealloc_network(&b);
	dealloc_network(&c);
	//print_layers(&c);
	*/
	// Create a pool of NN's with randomly initialized weights
	
	for(int i = 0; i < POOL_SIZE; i++){
		pool[i] = createMLP(INPUT_DIMENSIONS, 20, 10, OUTPUT_DIMENSIONS);
	}
	float avg_fitness = 0;
	float last_fitness = 0;
	size_t pause = 1;	
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
		for(int j = 0; j < POOL_SIZE && (!(i%pause)); j++){
//			printf("%d: %f\n", j, pool[j].performance);
		}
		for(int j = POOL_SIZE/2; j < POOL_SIZE; j++){
//			printf("Killing %f due to subpar performance (%f)\n", sumweights(pool[j].output), pool[j].performance);
			dealloc_network(&pool[j]);
			MLP *parent1 = &pool[rand()%(POOL_SIZE/2)];
			MLP *parent2 = &pool[rand()%(POOL_SIZE/2)];
//			printf("crossbreed start\n");
			pool[j] = crossbreed(parent1, parent2);
//			printf("crossbreed done\n");
			mutate(pool[j].output, LEARNING_RATE, MUTATION_RATE);
//			printf("done\n");
		}
		float fitness = pool[0].performance;
		avg_fitness += fitness;
//		float improvement = 100 * ((last_fitness/fitness) - 1);
		//Debug stuff
		if(!(i % pause)){
			float similarity = similarity_score(&pool[0], &pool[POOL_SIZE-1]);
			printf("CURRENTLY ON GENERATION %d, best fitness: %5.3f, avg %6.4f, similarity between best and worst network: %3.2f%%       \r", i, pool[0].performance, avg_fitness/i, 100*similarity);
//			getchar();
		}	
		if(avg_fitness/i > 0){
			printf("\nFitness threshold reached at %f after %d iterations.\n", avg_fitness/i, i);
			exit(0);
		}
	}
}
