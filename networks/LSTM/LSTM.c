/* Author: Jonah Siekmann
 * 10/3/2018
 * This is an attempt to write a Long Short-Term Memory network (LSTM) framework. 
 * I elected to rewrite large chunks of code instead of re-use MLP/RNN code, because
 * that would have resulted in very convoluted and difficult to understand (and write) code.
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
	n.cells = NULL;
	n.performance = 0;
	n.plasticity = 0.05;
	return n;
}

static Cell_Layer create_cell_layer(size_t size, Layer *input, Layer *output){
	Cell_Layer *layer = malloc(size

}
/*
 */
LSTM createLSTM(size_t input_dimension, size_t cells, size_t output_dimension){
	LSTM n = initLSTM();
	n.input = create_layer(input_dimension + cells, NULL);
	n.output = create_layer(output_dimension, NULL);
	n.hidden = create_cell_layer(cells, n.input, n.output);
	return n;
}
