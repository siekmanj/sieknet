#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include "lstm.h"


#define CREATEONEHOT(name, size, index) float name[size]; memset(name, '\0', size*sizeof(float)); name[index] = 1.0;

typedef uint8_t bool;

size_t HIDDEN_LAYER_SIZE = 100;
size_t NUM_EPOCHS = 5;
size_t ASCII_RANGE = 96; //96 useful characters in ascii: A-Z, a-z, 0-9, !@#$%...etc

float LEARNING_RATE = 0.01;

/*
 * This file is for training an LSTM character-by-character on any text (ascii) file provided.
 */
 
//Calling convention: ../bin/char [load/new] [path_to_modelfile] [path_to_txt_file]


static inline char int2char(int i){
	//printf("CONVERTING %d, SHOULD BECOME %d-32 (%c)\n", i, i, i);
	if(i==95) return '\n';
	if(i>95) return ' ';
	return i+32;
}
static inline int char2int(char c){
	if(c=='\n') return 95;
	int intval = c-32;
	if(intval < 0) return 0;
	if(intval > 95) return 0;
	return intval;
}
void bad_args(char *s, int pos){
	printf("bad argument '%s' for argument %d\n", s, pos);
	exit(1);
}

int train(LSTM *n, char *datafile, size_t num_epochs, float learning_rate){
	/* Begin training */
	wipe(n);
	n->learning_rate = learning_rate;
	n->stateful = 1;
	n->seq_len = 25;

	FILE *fp = fopen(datafile, "rb");
	fseek(fp, 0, SEEK_END);
	size_t datafilelen = ftell(fp);
	fclose(fp);

	for(int i = 0; i < num_epochs; i++){
		FILE *fp = fopen(datafile, "rb");
		size_t counter = 0;
		size_t training_iterations = 500;
		float avg_cost = 0;
		char input_char = fgetc(fp);
		do{
			char label = fgetc(fp);
			//printf("feeding %d from %c in to network\n", char2int(input_char), input_char);
			CREATEONEHOT(x, ASCII_RANGE, char2int(input_char));
			CREATEONEHOT(y, ASCII_RANGE, char2int(label));
			
			lstm_forward(n, x);
			float c = n->cost(n, y);
			lstm_backward(n);

			if(int2char(label) == '\n') printf("\n");
			else if(n->guess == char2int(label)) printf("%c", int2char(n->guess));
			else printf("_");
			//printf("guess %d vs real %d\n", n->guess, char2int(label));
			//printf("input: %d, %c (%d and %c)\n", input_char, input_char, char2int(input_char), char2int(int2char(input_char)));

			avg_cost += c;

			if(!(counter++) % (training_iterations*n->seq_len)){
				wipe(n);
				printf("Epoch %5.2f%% complete, avg cost %f.\n", 100 * ((float)counter)/datafilelen, avg_cost/counter);
				printf("%lu character sample from lstm below:\n", training_iterations);
				int seed = rand() % 95;
				for(int j = 0; j < training_iterations; j++){
					CREATEONEHOT(tmp, ASCII_RANGE, seed);
					lstm_forward(n, tmp);
					printf("%c", int2char(n->guess));
					seed = n->guess;
				}
				printf("\nResuming training...\n");
				sleep(2);
			}
			input_char = label;
		}
		while(input_char != EOF);
		fclose(fp);
	}
}

int main(int argc, char** argv){

	if(argc < 4){ printf("must provide %d args.\nexample usage: ./char [new/load] [path_to_modelfile] [path_to_datafile]\n", 3); exit(1);}

	srand(time(NULL));
	setbuf(stdout, NULL);

	bool newlstm;
	if(!strcmp(argv[1], "load")) newlstm = 0;
	else if(!strcmp(argv[1], "new")) newlstm = 1;
	else bad_args(argv[1], 0);
	
	char *modelfile = argv[2];
	char *datafile = argv[3];
	printf("modelfile is: '%s', datafile is: '%s'\n", modelfile, datafile);

	FILE *fp = fopen(datafile, "rb");
	size_t datafilelen;
	if(!fp){ printf("Could not open datafile '%s' - does it exist?\n", datafile); exit(1);}
	else{
	}
	fclose(fp);

	LSTM n;
	if(newlstm) n = create_lstm(ASCII_RANGE, HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE, ASCII_RANGE);
	else{
		fp = fopen(modelfile, "rb");
		if(!fp){ printf("Could not open modelfile '%s' - does it exist?\n", modelfile); exit(1);}
		n = load_lstm(modelfile);
	}
	save_lstm(&n, modelfile);

	train(&n, datafile, NUM_EPOCHS, LEARNING_RATE);
	
	
	//lstm_forward(&n, x);
	//n.cost(&n, y);
	//lstm_backward(&n);
	/*


	printf("range: %lu\n", ASCII_RANGE);
	for(int i = 0; i < ASCII_RANGE; i++){
		printf("index %d has: '%c'\n", i, int2char(i));
	}*/


}
