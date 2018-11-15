/* Author: Jonah Siekmann
 * This is an example of how you might use a genetic algorithm rather than backpropagation
 * to train a neural network.
 * For simplicity's sake, we'll be using the same problem as in binary.c
 */

#include "MLP.h"
#include "GA.h"
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
