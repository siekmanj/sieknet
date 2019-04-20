/* Jonah Siekmann
 * 7/24/2018
 * In this file are some tests I've done with the network.
 */
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <lstm.h>
#include <rnn.h>
#include <optimizer.h>

#define ARR_FROM_GPU(name, gpumem, size) float name[size]; memset(name, '\0', size*sizeof(float)); check_error(clEnqueueReadBuffer(get_opencl_queue0(), gpumem, 1, 0, sizeof(float) * size, name, 0, NULL, NULL), "error reading from gpu (ARR_FROM_SIEKNET_USE_GPU)");

/*
 * This is a simple demonstration of something a generic neural network would find difficult.
 * The network is trained to match a pattern which outputs a number n, n times.
 * For instance, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4 .... and so on.
 */

size_t MAX_ITERATIONS = 10000;

double LR = 0.00001;

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
  srand(1);
  setbuf(stdout, NULL);
  
  RNN n = create_rnn(10, 5, 10);

  Momentum o = create_optimizer(Momentum, n);
  o.alpha = LR;
  o.beta = 0.99;

  float cost = 0;
  float cost_threshold = 0.5;
  int count = 0;

  for(int epoch = 0; epoch < MAX_ITERATIONS; epoch++){ //Train for 1000 epochs.
    size_t len = sizeof(data)/sizeof(data[0]);
    n.seq_len = len;
    rnn_wipe(&n);

    for(int i = 0; i < len; i++){ //Run through the entirety of the training data.

      //Make a one-hot vector and use it to set the activations of the input layer
      float one_hot[10];
      memset(one_hot, '\0', 10*sizeof(float));
      one_hot[data[i]] = 1.0;

      int label = data[(i+1) % len]; //Use the next character in the sequence as the label	
      float expected[10];
      memset(expected, '\0', 10*sizeof(float));
      expected[label] = 1.0;

      rnn_forward(&n, one_hot);
      float c = rnn_cost(&n, expected);
      rnn_backward(&n);

      if(!n.t){
        o.step(o);
      }

      cost += c;

      int guess = n.guess;

      count++;	

      if(!(epoch % 10) && !i){
        printf("iter %d: label: %d, input: %d, output: %d, cost: %5.2f, avgcost: %5.2f, correct: %d\n", epoch, label, data[i], guess, c, cost/count, guess == label);
      }
    }

    if(epoch == MAX_ITERATIONS-1 || cost/count < cost_threshold){
      printf("\nCost threshold %1.2f reached in %d iterations\n", cost_threshold, epoch);
      printf("Running sequence:\n");
      rnn_wipe(&n);
      printf("1, ");
      int input = 1;
      for(int i = 0; i < len; i++){
        float one_hot[10];
        memset(one_hot, '\0', 10*sizeof(float));
        one_hot[input] = 1.0;
        rnn_forward(&n, one_hot);
        printf("%d, ", n.guess);
        input = n.guess;
      }
      printf("\n");
      save_rnn(&n, "./model/test.rnn");
      exit(0);
    }
  }
}
