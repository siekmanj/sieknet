#ifndef LSTM_H
#define LSTM_H

#include <stdlib.h>
#include <stdio.h>
#include "RNN.h"


// some magic to allow arbitrary numbers of parameters
#define createRNN(...) rnn_from_arr((size_t[]){__VA_ARGS__}, sizeof((size_t[]){__VA_ARGS__})/sizeof(size_t))

typedef MLP LSTM;

LSTM lstm_from_arr(size_t arr[], size_t size);
LSTM loadLSTMFromFile(const char *filename);

void feedforward_forget(LSTM *n);

#endif
