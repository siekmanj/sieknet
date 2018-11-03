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

float evaluate_fitness(MLP *n){
	
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
		float ans = bit0 * pow(2, 0) + bit1 * pow(2, 1) + bit2 * pow(2, 2) + bit3 * pow(2, 3);

		float arr[4] = {bit0, bit1, bit2, bit3}; //Input array (1 bit per input)
		MLP n = createMLP(INPUT_DIMENSIONS, 2, 2, OUTPUT_DIMENSIONS);
		setInputs(&n, arr);

		feedforward(&n); //Calculate outputs and run backprop
		gradients_wrt_outputs(n.output);
		
		//Debug stuff
		if(!(i % 1000)){
			printf("CURRENTLY ON EXAMPLE %d\n", i);
			printOutputs(n.output);
			printWeights(n.output);
			printActivationGradients(n.output);
			printActivationGradients(n.output->input_layer);
//			printf("Label %2d, guess %2d, Cost: %5.3f\n\n(ENTER to continue, CTRL+C to quit)\n", (int)ans, bestGuess(&n), cost);
			getchar();
		}	
	}
}
