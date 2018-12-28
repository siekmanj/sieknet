#include "LSTM.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define INPUT_DIM 10
#define NUMCELLS 15

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

//void forward


int main(){
	LSTM n = createLSTM(INPUT_DIM, NUMCELLS, NUMCELLS, INPUT_DIM, 0);
	/*
	for(int i = 0; i < 3; i++){
		float x1[] = {0.1, 1.5};
		forward(&n, x1);
		cost(&n, x1);
		backward(&n);
	}
	*/
	float cost = 0;
	float cost_threshold = 0.5;
	size_t count = 0;
	for(int epoch = 0; epoch < 100000; epoch++){ //Train for 1000 epochs.
		size_t len = sizeof(data)/sizeof(data[0]);

		for(int i = 0; i < len; i++){ //Run through the entirety of the training data.
			printf("\r");

			//Make a one-hot vector and use it to set the activations of the input layer
			float one_hot[INPUT_DIM];
			memset(one_hot, '\0', INPUT_DIM*sizeof(float));
			one_hot[data[i]] = 1.0;

			int label = data[(i+1) % len]; //Use the next character in the sequence as the label	

			float expected[INPUT_DIM];
			memset(expected, '\0', INPUT_DIM*sizeof(float));
			expected[label] = 1.0;

			
//			float c = step(&n, one_hot, expected);
			forward(&n, one_hot);
			float c = quadratic_cost(&n, expected);
			backward(&n);
			
			cost += c;

			count++;	
			if(count%100 == 0){
				printf("%d: cost: %3.2f, avg: %5.4f\n", count, c, cost/count);
				printf("desired output (label %d): [", label);
				for(int j = 0; j < INPUT_DIM; j++){
					printf("%4.3f", expected[j]);
					if(j < INPUT_DIM-1) printf(", ");
					else printf("]\n");
				}
				printf("actual output: [");
				for(int j = 0; j < INPUT_DIM; j++){
					printf("%4.3f", n.tail->output[j]);
					if(j < INPUT_DIM-1) printf(", ");
					else printf("]\n");
				}


				getchar();
			}
		
//			printf("label: %d, input: %d, output: %d, cost: %5.2f, avgcost: %5.2f, correct: %d", label, data[i], bestGuess(&n), c, cost/count, bestGuess(&n) == label);
		}

//	  if(cost/count < cost_threshold){
//			printf("\nCost threshold %1.2f reached in %d iterations\n", cost_threshold, count);
//			exit(0);
//		}
	}
}
