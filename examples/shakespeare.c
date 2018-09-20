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
	setbuf(stdout, NULL);
	RNN n;

	if(getchar() == 'n'){
		printf("creating new network...\n");
		n = createRNN(strlen(alphabet), 350, 350, 350, strlen(alphabet));
	}else{
		printf("loading network from file...\n");
		RNN n = loadRNNFromFile("../saves/rnn_sonnets_experimental.rnn");
	}
	n.plasticity = 0.01;	

	int count = 0;
	int debug = 0;
	int epochs = 1000;
	int debug_interval = 5000;
	for(int i = 0; i < epochs; i++){
		FILE *fp = fopen("../shakespeare/sonnets.txt", "rb");
	
		char input_character = ' ';
		char label = label_from_char(fgetc(fp), alphabet);
		float cost = 0;
		float lastavgcost = 10000000;
		float epochcost = 0;
		float epochcount = 0;
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
			if(count % 2000 == 0){
				epochcount += count;
				epochcost += cost;
				//printf("most recent activation gradients for output:\n");
				//printActivationGradients(n.output);
				printf("\n\n****\nlatest cost: %f vs previous cost: %f vs epoch avg cost:%f\n****\n\n", cost/count, lastavgcost, epochcost/epochcount);
				Neuron *output_neuron = &n.output->neurons[bestGuess(&n)];
				printf("output neuron (%d) has gradient %f, dActivation %f, output %f, \'%c\'\n\n", bestGuess(&n), output_neuron->activationGradient, output_neuron->dActivation, output_neuron->activation, alphabet[bestGuess(&n)]);
				lastavgcost = cost/count;
				cost = 0;
				count = 0;
			}
		}
	
		while(label != EOF);

		fclose(fp);
		printf("\n\n***********\nepoch %d concluded, avgcost: %f.\n************\n\n", i, epochcost/epochcount);
		saveRNNToFile(&n, "../saves/rnn_sonnets_experimental.rnn"); 

		char input = alphabet[rand()%(strlen(alphabet)-1)];
		printf("Sample from input '%c':\n", input);
		for(int i = 0; i < 2000; i++){
			float input_vector[strlen(alphabet)];		
			make_one_hot(input, alphabet, input_vector);
			
			setOneHotInput(&n, input_vector);
			
			feedforward_recurrent(&n);
	
			input = alphabet[bestGuess(&n)];
			printf("%c", input);
		}
		if(i % 30 == 0 && i != 0) getchar(); //wait for user input to continue after 30 epochs
	}
}
