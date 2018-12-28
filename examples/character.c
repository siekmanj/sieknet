#define UNROLL_LENGTH 100
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include "RNN.h"
#include "LSTM.h"


/*
 * This is a simple demonstration of the recurrent neural network.
 * The network is trained character-by-character to output a simple sentence (you are free to provide your own sentence and alphabet).
 */
const char *training = "A horse walked into a bar and said 'Can I have a drink please?'. The bartender said 'Hay, why not.'"; 
const char *alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz,.?' "; //possible inputs/outputs


/*
 * Description: This is a function that uses an input character to create a one-hot input vector.
 * inpt: the character whose position in alphabet will be used to set a single element to 1.0 in dest.
 * alphabet: the list of all possible inputs and outputs of the network (needs to be the same length as input/output layer of network)
 * dest: the output float array.
 */
void make_one_hot(char inpt, const char *alphabet, float *dest){
	for(int i = 0; i < strlen(alphabet); i++){
		if(inpt == alphabet[i]){
			dest[i] = 1.0;
		}
		else{
			dest[i] = 0.0;
		}
	}
	return;
}

/*
 * Description: This function turns a character into a label using that character's position in an alphabet.
 * inpt: the character whose position in alphabet will be used to return a label.
 * alphabet: the list of all possible inputs and outputs of the network (needs to be the same length as input/output layer of network)
 */
int label_from_char(char inpt, const char *alphabet){
	for(int i = 0; i < strlen(alphabet); i++){
		if(alphabet[i] == inpt) return i;
	}
	return EOF;
}

int main(void){

//	RNN n = createRNN(strlen(alphabet), 50, 50, strlen(alphabet)); //Create a network with a sufficiently large input & output layer, and a 40-neuron hidden layer.
	LSTM n = createLSTM(strlen(alphabet), 50, strlen(alphabet), 0);
	RNN x = createRNN(strlen(alphabet), 50, strlen(alphabet));
	n.plasticity = 0.05; //The network seems to perform best with a learning rate of around 0.1.
	x.plasticity = 0.05;
	/*
  Layer *current = n.input;

	//This is an experimental feature I am working on.	
	while(current != NULL){
    if(!(current == n.input || current == n.output)){
      current->squish = hypertan; //assigns this layer's squish function pointer to the tanh activation function
			current->dropout = 0.05; //approximately 5% of this layer's neurons will randomly not fire (dropout)
    }
		current = current->output_layer;
  }
	*/
	int epochs = 1000;
	float cost_threshold = 0.01;
	float epoch_cost_lstm = 0;
	float avg_cost_lstm = 0;

	float epoch_cost_rnn = 0;
	float avg_cost_rnn = 0;

	for(int i = 0; i < epochs*strlen(training); i++){ //Run the network for 1000 epochs.
		int input_idx = i % strlen(training);
		int label_idx = (i+1) % strlen(training);

		//Create a one-hot input vector.
		float input_one_hot[strlen(alphabet)];
		make_one_hot(training[input_idx], alphabet, input_one_hot);

		float expected[strlen(alphabet)];
		make_one_hot(training[label_idx], alphabet, expected);

		setOneHotInput(&x, input_one_hot); 	

		int label = label_from_char(training[label_idx], alphabet); //Create a label from the next character.
//		float c = step(&n, label); //Perform feedforward & backpropagation.
//		float c = step(&n, input_one_hot, expected);
		forward(&n, input_one_hot);
		float c = quadratic_cost(&n, expected);
		backward(&n);

		float cx = step(&x, label);
		
		epoch_cost_lstm += c/strlen(training);
		avg_cost_lstm += c;
		epoch_cost_rnn += cx/strlen(training);
		avg_cost_rnn += cx;
		int guess = 0;
		for(int i = 0; i < strlen(alphabet); i++){
			if(n.tail->output[i] > n.tail->output[guess]) guess = i;
		}

//		printf("%c", alphabet[guess]);

		if(i % strlen(training) == 0 && i != 0){
			printf("\nLSTM cost: %f, avgcost: %f. RNN cost: %f, avgcost %f\r", epoch_cost_lstm, avg_cost_lstm/i, epoch_cost_rnn, avg_cost_rnn/i);
			if(cost_threshold > epoch_cost_lstm){
				printf("Cost threshold of %f reached (%f) in %d examples.\n", cost_threshold, epoch_cost_lstm, i);
				exit(0);
			}
			epoch_cost_lstm = 0;
			epoch_cost_rnn = 0;
		}
	}

	//The below code prints sample output, where the network's guess is fed back in as input.
	/*
	char in = 'h';
	printf("%c", in);
	for(int i = 0; i < 100; i++){
		float input_one_hot[strlen(alphabet)];
		make_one_hot(in, alphabet, input_one_hot);

		setOneHotInput(&n, input_one_hot); 
		feedforward_recurrent(&n);
		printf("%c", alphabet[bestGuess(&n)]);
		in = alphabet[bestGuess(&n)];	
	}*/
}
