/* Jonah Siekmann
 * 7/24/2018
 * In this file are some tests I've done with the network.
 */
#include <stdio.h>
#include <time.h>
#include <string.h>
#include "lstm.h"
#include "optimizer.h"


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
	srand(time(NULL));
	LSTM n = create_lstm(10, 45, 10); //Create a network with 4 layers. Note that it's important that the input and output layers are both 10 neurons large.
	//n.learning_rate = 0.01;
  //Momentum o = create_optimizer(Momentum, n);
  //o.alpha = 0.0001;
  //o.beta = 0.99;
	//SGD o = create_optimizer(SGD, n);
 	//o.learning_rate = 0.05;
	Momentum o = create_optimizer(Momentum, n);
	o.alpha = 0.001;
	//SGD o1 = init_SGD(n.params, n.param_grad, n.num_params - n.output_layer.num_params);
	//SGD o2 = init_SGD(n.output_layer.params, n.output_layer.param_grad, n.output_layer.num_params);

	float cost = 0;
	float cost_threshold = 0.4;
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

			//printf("t: %lu, mlp paramgrad: %f, lstm paramgrad: %f\n", n.t, n.output_layer.param_grad[0], n.param_grad[10]);
			//o2.step(o2);
      //if(!n.t && i) o1.step(o1);
			if(!n.t) o.step(o);

			cost += c;

			int guess = n.output_layer.guess;

			count++;	
		
			//if(!(epoch % 1000) && label != data[i]){
				printf("label: %d, input: %d, output: %d, cost: %5.2f, avgcost: %5.2f, correct: %d\n", label, data[i], guess, c, cost/count, guess == label);
			//}
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
			exit(0);
		}
	}
}
