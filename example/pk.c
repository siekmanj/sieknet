#include <stdlib.h>
#include <stdio.h>
#include <lstm.h>

#define FEATURES 103
#define ONEHOTS  4

#define ROWLEN (FEATURES + ONEHOTS)

static float ***get_seqs(const char *filename){

}

static float free_seqs(float ***seq){

}

int main(){
  LSTM n = create_lstm(ROWLEN, 50, 1);
  SGD o = create_optimizer(SGD, n);

  int converged = 0;
  do {
    float **seq = get_seq(fp, &n.seq_len);
    
    for(int i = 0; i < n.seq_len; i++){
      float *x = seq[i];
      lstm_forward(&n, x);
      float c = lstm_cost(&n, y);
      lstm_backward(&n, x);
    }
    o->step(o);

  } while(!converged);

  return 0;
}

