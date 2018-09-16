#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include "RNN.h"

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

int label_from_char(char inpt, const char *alphabet){
	for(int i = 0; i < strlen(alphabet); i++){
		if(alphabet[i] == inpt) return i;
	}
	printf("ERRRRRRR\n");
	while(1);
	return -1;
}

int main(void){
	const char *training = " hello world what is up"; //desired sentence
	const char *inputs = "helowrdahtisup "; //possible inputs/outputs

	RNN n = createRNN(strlen(inputs), 20, strlen(inputs));

	for(int epoch = 0; epoch < 1000; epoch++){
		char output[strlen(inputs)];
		memset(output, '\0', strlen(inputs));
		float cost = 0;	
		for(int i = 0; i < strlen(training) - 1; i++){
			float input_one_hot[strlen(inputs)];
			make_one_hot(training[i], inputs, input_one_hot);

			setOneHotInput(&n, input_one_hot); 
		
			int label = label_from_char(training[i+1], inputs);	
			
			float c = step(&n, label);
			cost += c;
			output[i] = inputs[bestGuess(&n)];
		}
		cost /= strlen(training);
		if(epoch % 150 == 0){
			printf("output: %s\n", output);
			getchar();
		}
	}
	char in = ' ';
	for(int i = 0; i < 100; i++){
		float input_one_hot[strlen(inputs)];
		make_one_hot(in, inputs, input_one_hot);

		setOneHotInput(&n, input_one_hot); 
		feedforward_recurrent(&n);
		printf("%c", inputs[bestGuess(&n)]);
		in = inputs[bestGuess(&n)];	
	}
}
