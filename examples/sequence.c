/* Jonah Siekmann
 * 7/24/2018
 * In this file are some tests I've done with the network.
 */
#include <stdio.h>
#include <string.h>
#include "RNN.h"

/*
 * This is a simple demonstration of something a generic neural network would find 
 * difficult, but a RNN can do fairly easily.
 * The network is trained to match a pattern which outputs a number n, n times.
 * For instance, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4 .... and so on.
 */

int data[] = {
	1,
	2, 2,
	3, 3, 3,
	4, 4, 4, 4,
	5, 5, 5, 5, 5,
	6, 6, 6, 6, 6, 6,
	7, 7, 7, 7, 7, 7, 7,
	8, 8, 8, 8, 8, 8, 8, 8,
	9, 9, 9, 9, 9, 9, 9, 9, 9,
};

int main(void){
	srand(time(NULL));
	printf("Hello!\n");
	RNN n = createRNN(10, 15, 15, 10); //Create a network with 4 layers. Note that it's important that the input and output layers are both 10 neurons large.
	n.plasticity = 0.05;

	float cost = 0;
	int count = 0;

	//This is an experimental activation function I am testing.	
	Layer *current = n.input;
	while(current != NULL){
    if(!(current == n.input || current == n.output)){
      current->squish = hypertan; //assigns this layer's squish function pointer to the tanh activation function
    }
		current = current->output_layer;
  }
	for(int epoch = 0; epoch < 10000; epoch++){ //Train for 1000 epochs.
		size_t len = sizeof(data)/sizeof(data[0]);

		for(int i = 0; i < len; i++){ //Run through the entirety of the training data.

			//Make a one-hot vector and use it to set the activations of the input layer
			float one_hot[10];
			memset(one_hot, '\0', 10*sizeof(float));
			one_hot[data[i]] = 1.0;
			setOneHotInput(&n, one_hot); 
			
			int label = data[(i+1) % len]; //Use the next character in the sequence as the label	
			float c = step(&n, label); //Perform feedforward and backprop.
			cost += c;

			count++;	
		
			printf("label: %d, input: %d, output: %d, cost: %5.2f, avgcost: %5.2f, correct: %d\n", label, data[i], bestGuess(&n), c, cost/count, bestGuess(&n) == label);
		}
		if(cost/count < 0.5){
			printf("Cost threshold of 0.5 reached in %d iterations.\n", count);
			exit(0);
		}
	}
}
