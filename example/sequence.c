/* Jonah Siekmann
 * 7/24/2018
 * In this file are some tests I've done with the network.
 */
#include <stdio.h>
#include <time.h>
#include <string.h>
#include "lstm.h"
#include "optimizer.h"

#define PRINTLIST(name, len) printf("printing %s: [", #name); for(int xyz = 0; xyz < len; xyz++){printf("%5.4f", name[xyz]); if(xyz < len-1) printf(", "); else printf("]\n");}

/*
 * This is a simple demonstration of something a generic neural network would find difficult.
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
	//srand(time(NULL));
	srand(1);
	LSTM n = create_lstm(10, 4, 10); //Create a network with 4 layers. Note that it's important that the input and output layers are both 10 neurons large.
	//LSTM n = load_lstm("./model/test.lstm");
	Momentum o = create_optimizer(Momentum, n);
 	o.alpha = 0.005;
	o.beta = 0.95;

	float cost = 0;
	float cost_threshold = 1.0;
	int count = 0;

	for(int epoch = 0; epoch < 100000; epoch++){ //Train for 1000 epochs.
		size_t len = sizeof(data)/sizeof(data[0]);
    n.seq_len = len;
		wipe(&n);
		for(int i = 0; i < len; i++){ //Run through the entirety of the training data.

			//Make a one-hot vector and use it to set the activations of the input layer
			float one_hot[10];
			memset(one_hot, '\0', 10*sizeof(float));
			one_hot[data[i]] = 1.0;

			int label = data[(i+1) % len]; //Use the next character in the sequence as the label	
			float expected[10];
			memset(expected, '\0', 10*sizeof(float));
			expected[label] = 1.0;
			
			lstm_forward(&n, one_hot);
			float c = lstm_cost(&n, expected);
			lstm_backward(&n);

			if(!n.t){
				o.step(o);
			}

			cost += c;

			int guess = n.guess;

			count++;	
		
			if(!(epoch % 10)){
				//printf("iter %d: label: %d, input: %d, output: %d, cost: %5.2f, avgcost: %5.2f, correct: %d\n", epoch, label, data[i], guess, c, cost/count, guess == label);
			}
		}

	  if(cost/count < cost_threshold){
			printf("\nCost threshold %1.2f reached in %d iterations\n", cost_threshold, count);
      printf("Running sequence:\n");
      wipe(&n);
      printf("1, ");
      int input = 1;
      for(int i = 0; i < len; i++){
        float one_hot[10];
        memset(one_hot, '\0', 10*sizeof(float));
        one_hot[input] = 1.0;

        lstm_forward(&n, one_hot);
        printf("%d, ", n.guess);
        input = n.guess;
      }
			printf("\n");
			save_lstm(&n, "./model/test.lstm");
			exit(0);
		}
	}
}
