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

size_t HIDDEN_LAYER_SIZE = 550;
size_t NUM_EPOCHS = 1;
size_t ASCII_RANGE = 96; //96 useful characters in ascii: A-Z, a-z, 0-9, !@#$%...etc

float LEARNING_RATE     = 0.01;
float LEARNING_BASELINE = 0.000005;
float LEARNING_DECAY = 0.5;

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

char *get_sequence(FILE *fp, size_t *size){
	size_t seq_len = 0;
	while(1){
		char tmp = fgetc(fp);
		//printf("GOT: %c (%d)\n", tmp);
		seq_len++;
		if(tmp==EOF){
			return NULL;
		}
		if(tmp == '\n')
			break;
				
	}
	fseek(fp, -(seq_len), SEEK_CUR);
	*size = seq_len+1;

	char *ret = (char*)malloc(seq_len*sizeof(char));
	for(int i = 0; i < seq_len; i++){
		ret[i] = fgetc(fp);
	}
  //ret[seq_len-1] = '\0';
  //printf("found sequence '%s'\n", ret);
	return ret;
}

int train(LSTM *n, char *modelfile, char *datafile, size_t num_epochs, float learning_rate){
	/* Begin training */
	float learning_schedule[num_epochs];
	for(int i = 0; i < num_epochs; i++)
    learning_schedule[i] = learning_rate * pow(LEARNING_DECAY, i) + LEARNING_BASELINE;

	n->learning_rate = learning_rate;
	n->stateful = 1;

	FILE *fp = fopen(datafile, "rb");
	fseek(fp, 0, SEEK_END);
	size_t datafilelen = ftell(fp);
	fclose(fp);

	for(int i = 0; i < num_epochs; i++){
		n->learning_rate = learning_schedule[i];
		FILE *fp = fopen(datafile, "rb");
		size_t training_iterations = 100;
		size_t sequence_counter = 0;
    size_t ctr = 0;
		float avg_cost = 0;
		float avg_seq_cost = 0;
		char *seq = get_sequence(fp, &n->seq_len);
    wipe(n);
		do{
			float seq_cost = 0;
			printf("(sequence len %lu): '", n->seq_len);
			char input_char = '\n';
			for(int j = 0; j < n->seq_len; j++){
				char label = seq[j];
				CREATEONEHOT(x, ASCII_RANGE, char2int(input_char));
				CREATEONEHOT(y, ASCII_RANGE, char2int(label));

				lstm_forward(n, x);
				float c = lstm_cost(n, y);
				lstm_backward(n);
			
				if(n->guess == char2int(label)) printf("%c", int2char(n->guess));
				else printf("_");
				//printf("%c", label);
				seq_cost += c;
        ctr++;

				input_char = label;
			}
			printf("'\n");
			avg_seq_cost += seq_cost / n->seq_len;
			avg_cost += seq_cost / n->seq_len;
			sequence_counter++;

			if(!(sequence_counter % (training_iterations))){
				wipe(n);
				float completion =((float)ctr/datafilelen);
				printf("\n***\nEpoch %d %5.2f%% complete, avg cost %f (learning rate %6.5f), avg seq cost %6.5f.\n", i, 100 * completion, avg_cost/sequence_counter, n->learning_rate, avg_seq_cost / training_iterations);
				printf("%lu character sample from lstm below:\n", 10*training_iterations);

				int seed = 95;
				for(int j = 0; j < training_iterations*10; j++){
					CREATEONEHOT(tmp, ASCII_RANGE, seed);
					lstm_forward(n, tmp);
					printf("%c", int2char(n->guess));
					seed = n->guess;
				}
				if(avg_seq_cost/training_iterations > avg_cost/sequence_counter){
					printf("\nWARNING: average sequence cost was HIGHER than epoch average - something is probably wrong!\n");
				}else{
					printf("\nautosaving '%s'\n", modelfile);
					save_lstm(n, modelfile);
				}
				printf("\n***\nResuming training...\n");
				avg_seq_cost = 0;
				sleep(1);
        avg_seq_cost = 0;
			}
			seq = get_sequence(fp, &n->seq_len);
		}
		while(seq);
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
	if(newlstm) n = create_lstm(ASCII_RANGE, HIDDEN_LAYER_SIZE, ASCII_RANGE);
	else{
		printf("loading '%s'\n", modelfile);
		fp = fopen(modelfile, "rb");
		if(!fp){ printf("Could not open modelfile '%s' - does it exist?\n", modelfile); exit(1);}
		n = load_lstm(modelfile);
	}
	printf("network has %lu params.\n", n.num_params);
	save_lstm(&n, modelfile);

	train(&n, modelfile, datafile, NUM_EPOCHS, LEARNING_RATE);
	printf("training finished! LSTM saved to '%s'\n", modelfile);

}
