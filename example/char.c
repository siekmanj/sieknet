#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <string.h>

#include <lstm.h>
#include <rnn.h>
#include <optimizer.h>

#define CREATEONEHOT(name, size, index) float name[size]; memset(name, '\0', size*sizeof(float)); name[index] = 1.0;

#define USE_MOMENTUM 

#if !defined(USE_RNN) && !defined(USE_LSTM)
#define USE_LSTM
#endif

#ifndef HIDDEN_LAYER_SIZE
#define HIDDEN_LAYER_SIZE 256
#endif

#ifndef LAYERS
#define LAYERS 3
#endif

#ifndef NUM_EPOCHS
#define NUM_EPOCHS 3
#endif

#ifndef SAMPLE_CHARS
#define SAMPLE_CHARS 1000
#endif

#ifndef LR
#define LR 5e-4
#endif

#ifdef USE_RNN
#define network_type rnn
#define NETWORK_TYPE RNN
#endif

#ifdef USE_LSTM
#define network_type lstm
#define NETWORK_TYPE LSTM
#endif

#define forward_(arch) arch ## _forward
#define forward(arch) forward_(arch)

#define cost_(arch) arch ## _cost
#define cost(arch) cost_(arch)

#define backward_(arch) arch ## _backward
#define backward(arch) backward_(arch)

#define wipe_(arch) arch ## _wipe
#define wipe(arch) wipe_(arch)

#define create_(arch) create_ ## arch
#define create(arch) create_(arch)

#define from_arr_(arch) arch ## _from_arr
#define from_arr(arch) from_arr_(arch)

#define save_(arch) save_ ## arch
#define save(arch) save_(arch)

#define load_(arch) load_ ## arch
#define load(arch) load_(arch)

typedef uint8_t bool;

size_t SEQ_LEN					 = 150;
size_t ASCII_RANGE			 = 96; //96 useful characters in ascii: A-Z, a-z, 0-9, !@#$%...etc
size_t SAMPLE_EVERY			 = 100;

float LEARNING_RATE			 = LR;
float MOMENTUM					 = 0.99;

/*
 * This file is for training an rnn character-by-character on any text (ascii) file provided.
 */

static inline char int2char(int i){
  //printf("CONVERTING %d, SHOULD BECOME %d-32 (%c)\n", i, i, i);
  if(i==95) return '\n';
  if(i>95) return ' ';
  return i+32;
}
static inline int char2int(char c){
  if(c=='\n') return 95;
  int intval = c-32;
  if(intval < 0) return 0;
  if(intval > 95) return 0;
  return intval;
}
void bad_args(char *s, int pos){
  printf("bad argument '%s' for argument %d\n", s, pos);
  exit(1);
}

char *get_sequence(FILE *fp, size_t *size){
  char *ret = (char*)malloc(*size*sizeof(char));
  for(int i = 0; i < *size; i++){
    ret[i] = fgetc(fp);
    if(ret[i] == EOF){
      *size = 0;
      return NULL;
    }
  }
  return ret;
}

struct timespec diff(struct timespec start, struct timespec end){
  struct timespec temp;
  if ((end.tv_nsec-start.tv_nsec)<0){
    temp.tv_sec = end.tv_sec-start.tv_sec-1;
    temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
  }else{
    temp.tv_sec = end.tv_sec-start.tv_sec;
    temp.tv_nsec = end.tv_nsec-start.tv_nsec;
  }
  return temp;
}

void sample(NETWORK_TYPE *n, size_t chars, char seed){
  wipe(network_type)(n);
  int input = char2int(seed);
  for(int i = 0; i < chars; i++){
    CREATEONEHOT(tmp, ASCII_RANGE, input);
    forward(network_type)(n, tmp);
    printf("%c", int2char(n->guess));
    input = n->guess;
  }
  printf("\n");
}

void train(NETWORK_TYPE *n, char *modelfile, char *datafile, size_t num_epochs, float learning_rate){
  /* Begin training */
  Momentum o = create_optimizer(Momentum, *n);
  o.alpha = learning_rate;
  o.beta = MOMENTUM;

  FILE *tmp = fopen(datafile, "rb");
  fseek(tmp, 0, SEEK_END);
  size_t datafilelen = ftell(tmp);
  fclose(tmp);

  float learning_schedule[] = {
    learning_rate * 1.0,
    learning_rate * 0.7, 
    learning_rate * 0.5, 
    learning_rate * 0.5, 
    learning_rate * 0.25, 
    learning_rate * 0.125, 
    learning_rate * 0.1,
    learning_rate * 0.1,
    learning_rate * 0.1,
    learning_rate * 0.05,
    learning_rate * 0.05,
    learning_rate * 0.05
  };
  float last_epoch_cost = 4.5;
  for(int i = 0; i < num_epochs; i++){

    n->seq_len	= SEQ_LEN;
    n->stateful = 1;
    o.alpha = learning_schedule[i];

    FILE *fp = fopen(datafile, "rb");
    fseek(fp, 0, SEEK_SET);

    size_t training_iterations = SAMPLE_EVERY;
    size_t sequence_counter = 0;

    size_t ctr = 0;

    float avg_cost = 0;
    float avg_seq_cost = 0;
    float seq_time = 0;
    float avg_seq_time = 0;

    wipe(network_type)(n);

    char *seq = get_sequence(fp, &n->seq_len);
    char input_char = '\n';
    do{
      struct timespec start, end;
      clock_gettime(CLOCK_REALTIME, &start);

      float completion = ((float)ctr/datafilelen);
      if(sequence_counter % training_iterations){
        float time_left = (1-completion) * (datafilelen / n->seq_len) * ((avg_seq_time / ((sequence_counter % training_iterations))));
        int hrs_left = (int)(time_left / (60*60));
        int min_left = ((int)(time_left - (hrs_left * 60 * 60))) / 60;
        printf("%3lu/%3lu | (%5.4f s, appr. %2dh %2dmin left) | epoch %2d %4.2f%% of %2ld| last %3lu seqs: %4.3f | epoch cost: %5.4f | previous epoch: %5.4f | lr: %7.6f\r",
            sequence_counter % training_iterations, 
            training_iterations,
            seq_time,
            hrs_left,
            min_left,
            i,
            100 * completion, 
            num_epochs,
            sequence_counter % training_iterations, 
            avg_seq_cost / (sequence_counter % training_iterations), 
            avg_cost/sequence_counter, 
            last_epoch_cost,
            o.alpha
            );
      }
      float seq_cost = 0;
      for(int j = 0; j < n->seq_len; j++){
        char label = seq[j];

        CREATEONEHOT(x, ASCII_RANGE, char2int(input_char));
        CREATEONEHOT(y, ASCII_RANGE, char2int(label));

        forward(network_type)(n, x);
        float c = cost(network_type)(n, y);
        backward(network_type)(n);

        if(!n->t){
          o.step(o);
        }

        seq_cost += c;
        ctr++;

        input_char = label;
      }
      avg_seq_cost += seq_cost / n->seq_len;
      avg_cost += seq_cost / n->seq_len;
      sequence_counter++;

      clock_gettime(CLOCK_REALTIME, &end);
      struct timespec elapsed = diff(start, end);
      seq_time = (double)elapsed.tv_sec + ((double)elapsed.tv_nsec) / 1000000000;
      avg_seq_time += seq_time;
      if(!(sequence_counter % (training_iterations))){
        printf("\n");
        wipe(network_type)(n);
        for(int i = 3; i > 0; i--){
          printf("Sampling from model in %d\r", i);
          sleep(1);
        }
        sample(n, SAMPLE_CHARS, '\n');
        if(isnan(avg_cost) || avg_seq_cost/training_iterations > avg_cost/sequence_counter){
          printf("\nWARNING: average sequence cost was HIGHER than epoch average - something is probably wrong!\n");
        }else{
          printf("\nautosaving '%s'\n", modelfile);
          save(network_type)(n, modelfile);
        }
        printf("\n***\nResuming training...\n");
        avg_seq_cost = 0;
        avg_seq_time = 0;
        wipe(network_type)(n);
      }
      free(seq);
      seq = get_sequence(fp, &n->seq_len);
    }
    while(seq && n->seq_len > 0);
    fclose(fp);
    last_epoch_cost = avg_cost / sequence_counter;
  }
}

int main(int argc, char** argv){

  if(argc < 4){ printf("%d args needed. Usage: ./char [new/load] [path_to_modelfile] [path_to_datafile]\n", 3); exit(1);}

  printf("   _____ ____________ __ _   ______________\n");
  printf("  / ___//  _/ ____/ //_// | / / ____/_	__/\n");
  printf("  \\__ \\ / // __/ / ,<  /  |/ / __/   / /   \n");
  printf(" ___/ // // /___/ /| |/ /|  / /___  / /    \n");
  printf("/____/___/_____/_/ |_/_/ |_/_____/ /_/	   \n");
  printf("																					 \n");
  printf("ascii-nn recurrent neural network interface.\n");

  srand(time(NULL));
  //srand(1);
  setbuf(stdout, NULL);

  size_t layersizes[LAYERS];
  layersizes[0] = ASCII_RANGE;
  for(int i = 1; i < LAYERS-1; i++)
    layersizes[i] = HIDDEN_LAYER_SIZE;
  layersizes[LAYERS-1] = ASCII_RANGE;

  bool newmodel;
  if(!strcmp(argv[1], "load")) newmodel = 0;
  else if(!strcmp(argv[1], "new")) newmodel = 1;
  else bad_args(argv[1], 0);

  char *modelfile = argv[2];
  char *datafile = argv[3];
  FILE *fp;

  NETWORK_TYPE n;
  if(newmodel){
    n = from_arr(network_type)(layersizes, LAYERS);
    printf("creating '%s'\n", modelfile);
  }else{
    printf("loading '%s'\n", modelfile);
    fp = fopen(modelfile, "rb");
    if(!fp){ printf("Could not open modelfile '%s' - does it exist?\n", modelfile); exit(1);}
    n = load(network_type)(modelfile);
    fclose(fp);
  }
  printf("network has %lu params.\n", n.num_params);
  save(network_type)(&n, modelfile);

  if(!strcmp(datafile, "sample")){
    printf("Sampling from model '%s' below:\n", modelfile);
    sample(&n, SAMPLE_CHARS, '\n');
    exit(0);
  }

  fp = fopen(datafile, "rb");
  if(!fp){ printf("Could not open datafile '%s' - does it exist?\n", datafile); exit(1);}
  fclose(fp);

  train(&n, modelfile, datafile, NUM_EPOCHS, LEARNING_RATE);
  printf("\ntraining finished! Saved to '%s'\n", modelfile);

}
