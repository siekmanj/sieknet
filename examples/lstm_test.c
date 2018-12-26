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
//	LSTM n = createLSTM(INPUT_DIM, 10);
	LSTM n = createLSTM(INPUT_DIM, NUMCELLS);
	MLP out = createMLP(NUMCELLS, INPUT_DIM);
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
/*
			one_hot[0] = 1.0;
			one_hot[1] = 2;
			Cell *cell = &n.cells[0];
			cell->input_nonl.weights[0] = 0.45;
			cell->input_nonl.weights[1] = 0.25;
			cell->input_nonl.weights[2] = 0.15;
			cell->input_nonl.bias = 0.2;

			cell->input_gate.weights[0] = 0.95;
			cell->input_gate.weights[1] = 0.8;
			cell->input_gate.weights[2] = 0.8;
			cell->input_gate.bias = 0.65;
			
			cell->forget_gate.weights[0] = 0.7;
			cell->forget_gate.weights[1] = 0.45;
			cell->forget_gate.weights[2] = 0.1;
			cell->forget_gate.bias = 0.15;

			cell->output_gate.weights[0] = 0.6;
			cell->output_gate.weights[1] = 0.4;
			cell->output_gate.weights[2] = 0.25;
			cell->output_gate.bias = 0.1;
*/			

			int label = data[(i+1) % len]; //Use the next character in the sequence as the label	

			float expected[INPUT_DIM];
			memset(expected, '\0', INPUT_DIM*sizeof(float));
			expected[label] = 1.0;

			printf("desired output: [");
			for(int j = 0; j < INPUT_DIM; j++){
				printf("%4.3f", expected[j]);
				if(j < INPUT_DIM-1) printf(", ");
				else printf("]\n");
			}
			
			float c = step(&n, one_hot, expected);
			
//			feedforward_forget(&n, one_hot);
//			float c = backpropagate_cells(&n, label);

//			printf("output (label %d, cost %5.3f): [", label, c);
//			for(int j = 0; j < INPUT_DIM; j++){
//				printf("%4.3f", n.cells[j].output);
//				if(j < INPUT_DIM-1) printf(", ");
//				else printf("]\n");
//			}
//			float c = step(&n, label); //Perform feedforward and backprop.
			cost += c;

			count++;	
			if(count%120 == 0){
				printf("%d: cost: %3.2f, avg: %5.4f\n", count, c, cost/count);
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
