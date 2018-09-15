#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include "RNN.h"

void make_one_hot(char inpt, const char *alphabet, float *dest){
	for(int i = 0; i < strlen(alphabet); i++){
		if(inpt = alphabet[i]){
			dest[i] = 1.0;
			return;
		}
		else dest[i] = 0.0;
	}
	printf("ERR NO HOT\n");
	return;
}

int main(void){
	const char *training = "hello world "; //desired sentence
	const char *inputs = "helowrld"; //possible inputs/outputs

	RNN n = createRNN(strlen(inputs), 15, strlen(inputs));
	printf("layers sizes: %lu, %d, %lu vs %lu, %lu, %lu\n", strlen(inputs), 15, strlen(inputs), n.input->size, ((Layer*)n.input->output_layer)->size, n.output->size); 	
	while(0){
		char output[strlen(inputs)];
		for(int i = 0; i < strlen(training) - 1; i++){
			float input_one_hot[strlen(inputs)];
			make_one_hot(training[i], inputs, input_one_hot);

			setInputs(&n, input_one_hot); 
			
			float c = descend(&n, training[i+1]);

			//output[i] = bestGuess(&n);	

			printf("c: %5.2f, output: %s\n", c, output);	
		}
	}
}
