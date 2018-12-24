#include "LSTM.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define INPUT_DIM 10

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



int main(){
	LSTM n = createLSTM(INPUT_DIM, 10);
	float cost = 0;
	float cost_threshold = 0.5;

	for(int epoch = 0; epoch < 100000; epoch++){ //Train for 1000 epochs.
		size_t len = sizeof(data)/sizeof(data[0]);

		for(int i = 0; i < len; i++){ //Run through the entirety of the training data.
			printf("\r");

			//Make a one-hot vector and use it to set the activations of the input layer
			float one_hot[INPUT_DIM];
			memset(one_hot, '\0', INPUT_DIM*sizeof(float));
			one_hot[data[i]] = 1.0;
			
			printf("input: [");
			for(int j = 0; j < INPUT_DIM; j++){
				printf("%4.3f", one_hot[j]);
				if(j < INPUT_DIM-1) printf(", ");
				else printf("]\n");
			}

			int label = data[(i+1) % len]; //Use the next character in the sequence as the label	
			feedforward_forget(&n, one_hot);

			printf("output: [");
			for(int j = 0; j < INPUT_DIM; j++){
				printf("%4.3f", n.cells[j].output);
				if(j < INPUT_DIM-1) printf(", ");
				else printf("]\n");
			}
			getchar();
//			float c = step(&n, label); //Perform feedforward and backprop.
//			cost += c;

//			count++;	
		
//			printf("label: %d, input: %d, output: %d, cost: %5.2f, avgcost: %5.2f, correct: %d", label, data[i], bestGuess(&n), c, cost/count, bestGuess(&n) == label);
		}

//	  if(cost/count < cost_threshold){
//			printf("\nCost threshold %1.2f reached in %d iterations\n", cost_threshold, count);
//			exit(0);
//		}
	}
}
