/* Jonah Siekmann
 * 7/24/2018
 * In this file are some tests I've done with the network.
 */
#include <stdio.h>
#include <time.h>
#include <string.h>
#include "LSTM.h"


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
	LSTM n = createLSTM(10, 15, 15, 10); //Create a network with 4 layers. Note that it's important that the input and output layers are both 10 neurons large.
	n.plasticity = 0.01;

	float cost = 0;
	float cost_threshold = 0.05;
	int count = 0;

	for(int epoch = 0; epoch < 100000; epoch++){ //Train for 1000 epochs.
		size_t len = sizeof(data)/sizeof(data[0]);
//		wipe(&n);
		for(int i = 0; i < len; i++){ //Run through the entirety of the training data.

			//Make a one-hot vector and use it to set the activations of the input layer
			float one_hot[10];
			memset(one_hot, '\0', 10*sizeof(float));
			one_hot[data[i]] = 1.0;

			int label = data[(i+1) % len]; //Use the next character in the sequence as the label	
			float expected[10];
			memset(expected, '\0', 10*sizeof(float));
			expected[label] = 1.0;
			
			forward(&n, one_hot);
//			float c = cross_entropy_cost(&n, expected);
			float c = quadratic_cost(&n, expected);
			backward(&n);

			cost += c;

			int guess = 0;
			for(int j = 0; j < 10; j++) if(n.tail->output[j] > n.tail->output[guess]) guess = j;

			count++;	
		
			if(!(epoch % 1000) && label != data[i]){
				printf("label: %d, input: %d, output: %d, cost: %5.2f, avgcost: %5.2f, correct: %d\n", label, data[i], guess, c, cost/count, guess == label);
				printf("outputs: [");
				float sum = 0;
				for(int j = 0; j < n.tail->size; j++){
					printf("%4.3f", n.tail->output[j]);
					if(j < n.tail->size-1) printf(", ");
					else printf("]\n");
					sum += n.tail->output[j];
				}
				printf("sum: %4.3f\n", sum);
			}
		}

	  if(cost/count < cost_threshold){
			printf("\nCost threshold %1.2f reached in %d iterations\n", cost_threshold, count);
			exit(0);
		}
	}
}
