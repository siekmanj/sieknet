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
	return EOF;
}

char *alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789,.;:?!-()[]'\"\n ";

int main(void){
	srand(time(NULL));
	RNN n;

	if(getchar() == 'n'){
		printf("creating new network...\n");
		n	= createRNN(strlen(alphabet), 200, 200, 200, strlen(alphabet));
	}else{
		printf("loading network from file...\n");
		RNN n = loadRNNFromFile("../saves/rnn_sonnets.rnn");
	}

	n.plasticity = 0.2;	

	int count = 0;
	int debug = 0;
	int epochs = 10;
	int debug_interval = 5000;

	for(int i = 0; i < epochs; i++){
		FILE *fp = fopen("../shakespeare/king_henry.txt", "rb");
	
		char input_character = ' ';
		char label = label_from_char(fgetc(fp), alphabet);
		float cost = 0;
		float lastavgcost = 10000000;

		do {
			
			float input_one_hot[strlen(alphabet)];
			make_one_hot(input_character, alphabet, input_one_hot);	
			setOneHotInput(&n, input_one_hot);
			
			float cost_local = step(&n, label);

			cost += cost_local;

			printf("%c", alphabet[bestGuess(&n)]);

			input_character = alphabet[label];
			count++;
			
			label = label_from_char(fgetc(fp), alphabet);
		}	
		while(label != EOF);

		fclose(fp);
		printf("\n\n***********\nepoch %d concluded, avgcost: %f.\n************\n\n", i, cost/count);
		saveRNNToFile(&n, "../saves/rnn_sonnets.rnn"); 

		printf("Sample:\n");
		char input = ' ';
		for(int i = 0; i < 1000; i++){
			float input_one_hot[strlen(alphabet)];
			make_one_hot(input, alphabet, input_one_hot);
			setOneHotInput(&n, input_one_hot);

			feedforward_recurrent(&n);
			input = alphabet[bestGuess(&n)];

			printf("%c", bestGuess(&n));
		}
		getchar();
	}
}
