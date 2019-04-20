#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <rnn.h>
#include <rnn.h>
#include <optimizer.h>

float PI = 3.14159;

int main(){
  RNN n = create_rnn(1, 1, 1);
  Momentum o = create_optimizer(Momentum, n);
  o.alpha = 0.001;

  n.seq_len = 100; // one sin period
  n.stateful = 1;

  n.output_layer.logistic = hypertan;
  n.cost_fn = quadratic;

  float avg_cost = 0;
  for(int i = 0; i < 10000 * n.seq_len; i++){
    float *x = malloc(sizeof(float));

    if(!(i % n.seq_len))
      *x = 0.0;
    else
      *x = 0.0;

    //float x[] = {((float)rand())/RAND_MAX};
    //float x[] = {0.0};
    //float x[] = {cos(2 * PI * i / (n.seq_len))};

    //float y[] = {0.5 * sin(2 * PI * i / (n.seq_len))};
    float y[] = {(float)(i&1) - 0.5};

    rnn_forward(&n, x);
    avg_cost += rnn_cost(&n, y);
    rnn_backward(&n);

    float guess = n.output[0];
    
    if(!n.t) o.step(o);

    if(!n.t){
      avg_cost /= 1000;
      printf("cost: %f             \r", avg_cost);
      avg_cost = 0;
      n.layers[0].loutput[0] = 1.0;
    }
    free(x);
  }
  printf("\n");

  for(int i = 0; i < 60; i++){
    float x[] = {0.0};
    rnn_forward(&n, x);
    float guess = n.output[0];
    printf("%f\n", guess);
  }


}
