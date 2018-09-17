#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include "RNN.h"

//tests an RNN on a sequence like 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, .....
//each number n is repeated n times

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
	RNN n = createRNN(10, 5, 10);
	int count = 0;
	float cost = 0;
	n.plasticity = 0.01;
	for(int epoch = 0; epoch < 1000000; epoch++){
		size_t len = sizeof(data)/sizeof(data[0]);
		for(int i = 0; i < len; i++){
			//Make a one-hot vector
			float one_hot[10];
			memset(one_hot, '\0', 10*sizeof(float));
			one_hot[data[i]] = 1.0;
			setOneHotInput(&n, one_hot); 
			
			int label = data[(i+1) % len];	
			float c = step(&n, label);
			cost += c;

			count++;	
		
				printf("label: %d, input: %d, output: %d, cost: %5.2f, avgcost: %5.2f, correct: %d\n", label, data[i], bestGuess(&n), c, cost/count, bestGuess(&n) == label);
			if(epoch > 10000){
			//	printf("input vector:\n");
			//	for(int j = 0; j < 10; j++) printf("%f, ", one_hot[j]); printf("\n");
			//	printf("input layer:\n");
			//	printOutputs(n.input);
			//	printf("hidden layer:\n");
			//	printOutputs((Layer*)n.input->output_layer);
			//	getchar();
			}
		}
		if(epoch % 100000 == 0){
			getchar();
		}
	}
}
