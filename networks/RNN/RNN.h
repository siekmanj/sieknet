#ifndef RNN_H
#define RNN_H

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "MLP.h"

// some magic to allow arbitrary numbers of parameters
#define createRNN(...) rnn_from_arr((size_t[]){__VA_ARGS__}, sizeof((size_t[]){__VA_ARGS__})/sizeof(size_t))


typedef MLP RNN;

RNN rnn_from_arr(size_t arr[], size_t size);
RNN loadRNNFromFile(const char *filename);

void setOneHotInput(RNN *n, float* arr);
void feedforward_recurrent(RNN *n);
void saveRNNToFile(RNN *n, char* filename);

float step(RNN *n, int label);

#endif
