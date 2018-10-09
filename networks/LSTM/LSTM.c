/* Author: Jonah Siekmann
 * 10/3/2018
 * This is an attempt to further build upon my MLP and RNN framework to create 
 * a Long Short-Term Memory network (LSTM). Every function with the static keyword
 * is meant for internal use only, and every function without can be called by the user.
 */

#include "LSTM.h"
#include <math.h>
#include <string.h>

/* 
 * [   ]   [   ]   
 * [INP] x [ H ]
 * [   ]   [   ]
 * remember gate: sigmoid of inputs and previous hidden state
 * candidate ltm: tanh of inputs and previous hidden state
 * 
 *
 */

/*
 * Description: Initializes a long short-term memory network object.
 */ 
static LSTM initLSTM(){
	LSTM n;
	n.input = NULL;
	n.output = NULL;
	n.performance = 0;
	n.plasticity = 0.05;
	return n;
}

/*
 * Description: a function called through a macro that allows creation of a network with any arbitrary number of layers.
 * arr: The array containing the sizes of each layer, for instance {28*28, 16, 10}.
 * size: The size of the array.
 */
LSTM lstm_from_arr(size_t arr[], size_t size){
	LSTM n = initLSTM();
	return n;
}


