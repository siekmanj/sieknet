
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include "RNN.h"

/*
 * This program demos the output of an RNN.
 */


char *alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789,.;:?!-()[]'\"\n ";

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
	srand(time(NULL));
	setbuf(stdout, NULL);

	char* filename = "../saves/rnn_sonnets_3x350.rnn"; //This is the network file that will be loaded, and the one that will be saved to.

	printf("Press ENTER to load %s (may take a while to load)\n", filename);
	getchar();

	printf("loading network from %s...\n", filename);
	RNN n = loadRNNFromFile(filename);

	char input = alphabet[rand()%(strlen(alphabet)-1)]; //Give the network a random character to start with.
	printf("Sample from input '%c':\n", input);


	//Get 2000 sample characters from the network
	for(int i = 0; i < 2000; i++){
		float input_vector[strlen(alphabet)];		
		make_one_hot(input, alphabet, input_vector);
		
		setOneHotInput(&n, input_vector);
		
		feedforward_recurrent(&n);

		input = alphabet[bestGuess(&n)];
		printf("%c", input);
	}
}	
