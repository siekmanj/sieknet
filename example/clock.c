#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <rnn.h>
#include <lstm.h>
#include <optimizer.h>

float PI = 3.14159;

int main(){
  srand(1);
  float best_cost = 1.0f;
  for(int l = 1; l < 10000; l+=100){
    float lr = (float)l/10000;
    for(int t = 4; t < 50; t++){

      LSTM n = create_lstm(0, 5, 1);
      n.seq_len = t;
      n.output_layer.logistic = hypertan;
      n.cost_fn = quadratic;
      SGD o = create_optimizer(SGD, n);
      o.learning_rate = lr;
      
      float avg_cost = 0;
      printf("trying %f and %lu\n", lr, n.seq_len);
      for(int i = 0; i < n.seq_len * 1000; i++){

        float y[] = {0.5 * cos((4 * PI * i )/(float)n.seq_len)};

        lstm_forward(&n, NULL);
        avg_cost += lstm_cost(&n, y);
        lstm_backward(&n);

        float guess = n.output[0];
        
        if(!n.t) o.step(o);
      }
      avg_cost /= n.seq_len * 1000;
      printf("resulting cost: %f\n", avg_cost);
      if(avg_cost < best_cost){
        printf("%f: new best config found: lr %f and sequence len %d\n", avg_cost, lr, n.seq_len);
        best_cost = avg_cost;
        for(int i = 0; i < 60; i++){
          float x[] = {0.0};
          lstm_forward(&n, x);
          float guess = n.output[0];
          printf("%f\n", guess);
        }
        getchar();
      }
      dealloc_lstm(&n);
    }
  }
<<<<<<< HEAD
  printf("\n");

  for(int i = 0; i < 60; i++){
    float x[] = {0.0};
    rnn_forward(&n, x);
    float guess = n.output[0];
    printf("%f\n", guess);
  }


=======
>>>>>>> 0db3d5b3dcf0ec71e9ba076d60d57ddfea559306
}
