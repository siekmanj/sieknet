#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <string.h>

#include <lstm.h>
#include <optimizer.h>


#define CREATEONEHOT(name, size, index) float name[size]; memset(name, '\0', size*sizeof(float)); name[index] = 1.0;

#define NEWLSEQ  0
#define STATEFUL 1
#define USE_MOMENTUM 1

#define CHAR_OUTPUT 0

typedef uint8_t bool;

size_t HIDDEN_LAYER_SIZE = 400;
size_t NUM_EPOCHS				 = 10;
size_t SEQ_LEN					 = 75;
size_t ASCII_RANGE			 = 96; //96 useful characters in ascii: A-Z, a-z, 0-9, !@#$%...etc
size_t SAMPLE_EVERY			 = 500;
size_t SAMPLE_CHARS			 = 1000;

float LEARNING_RATE			 = 0.0001;
float MOMENTUM					 = 0.99;

/*
 * This file is for training an LSTM character-by-character on any text (ascii) file provided.
 */
 
//Calling convention: ../bin/char [load/new] [path_to_modelfile] [path_to_txt_file/sample]


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
#if NEWLSEQ
	size_t seq_len = 0;
	while(1){
		char tmp = fgetc(fp);
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
#else
	char *ret = (char*)malloc(*size*sizeof(char));
	for(int i = 0; i < *size; i++){
		ret[i] = fgetc(fp);
		if(ret[i] == EOF){
			*size = 0;
			return NULL;
		}
	}
#endif
	return ret;
}

void sample(LSTM *n, size_t chars, char seed){
	wipe(n);
	int input = char2int(seed);
	for(int i = 0; i < chars; i++){
		CREATEONEHOT(tmp, ASCII_RANGE, input);
		lstm_forward(n, tmp);
		printf("%c", int2char(n->guess));
		input = n->guess;
	}
	printf("\n");
}

int train(LSTM *n, char *modelfile, char *datafile, size_t num_epochs, float learning_rate){
	/* Begin training */

#if USE_MOMENTUM
	Momentum o = create_optimizer(Momentum, *n);
	o.alpha = LEARNING_RATE;
	o.beta = MOMENTUM;
#else
	SGD o = create_optimizer(SGD, *n);
	o.learning_rate = LEARNING_RATE;
#endif


	FILE *tmp = fopen(datafile, "rb");
	fseek(tmp, 0, SEEK_END);
	size_t datafilelen = ftell(tmp);
	fclose(tmp);

	float last_epoch_cost = 4.5;
	for(int i = 0; i < num_epochs; i++){
		float learning_schedule[] = {
																 LEARNING_RATE * 1.0,
																 LEARNING_RATE * 0.7, 
																 LEARNING_RATE * 0.5, 
																 LEARNING_RATE * 0.5, 
																 LEARNING_RATE * 0.25, 
																 LEARNING_RATE * 0.125, 
																 LEARNING_RATE * 0.1,
																 LEARNING_RATE * 0.1,
																 LEARNING_RATE * 0.1,
																 LEARNING_RATE * 0.05,
																 LEARNING_RATE * 0.05,
																 LEARNING_RATE * 0.05
																};

		n->seq_len	= SEQ_LEN;
		n->stateful = STATEFUL;
#if USE_MOMENTUM
		o.alpha = learning_schedule[i];
#else
		o.learning_rate = learning_schedule[i];
#endif

		FILE *fp = fopen(datafile, "rb");
		fseek(fp, 0, SEEK_SET);

		size_t training_iterations = SAMPLE_EVERY;
		size_t sequence_counter = 0;

		size_t ctr = 0;

		float avg_cost = 0;
		float avg_seq_cost = 0;

		wipe(n);

		char *seq = get_sequence(fp, &n->seq_len);
		char input_char = '\n';
		do{
			float completion =((float)ctr/datafilelen);
#if CHAR_OUTPUT
			printf("(sequence len %lu): '", n->seq_len);
#else
			printf("sequence len %3lu, (%3lu)/(%3lu)), epoch %2d %5.2f%% complete. Cost over last %3lu sequences: %6.5f. Epoch cost: %6.5f. Previous epoch cost: %6.5f. Current lr: %7.6f\r",
						n->seq_len, 
						sequence_counter % training_iterations + 1 , 
						training_iterations, 100 * completion, 
						i+1,
						sequence_counter % training_iterations + 1, 
						avg_seq_cost / (sequence_counter % training_iterations + 1), 
						avg_cost/sequence_counter, 
						last_epoch_cost,
#if USE_MOMENTUM
						o.alpha
#else
						o.learning_rate
#endif
					 );
#endif
#if !(STATEFUL)
			input_char = '\n';
#endif
			float seq_cost = 0;
			for(int j = 0; j < n->seq_len; j++){
				char label = seq[j];

				CREATEONEHOT(x, ASCII_RANGE, char2int(input_char));
				CREATEONEHOT(y, ASCII_RANGE, char2int(label));

				lstm_forward(n, x);
				float c = lstm_cost(n, y);
				lstm_backward(n);

				if(!n->t) o.step(o);
			
#if CHAR_OUTPUT
				if(n->guess == char2int(label)) printf("%c", int2char(n->guess));
				else printf("_");
#endif

				seq_cost += c;
				ctr++;

				input_char = label;
			}
#if CHAR_OUTPUT	
			printf("'\n");
#endif
			avg_seq_cost += seq_cost / n->seq_len;
			avg_cost += seq_cost / n->seq_len;
			sequence_counter++;

			if(!(sequence_counter % (training_iterations))){
#if STATEFUL
				wipe(n);
#endif

#if CHAR_OUTPUT
				printf("\n***\nEpoch %d %5.2f%% complete, avg cost %6.5f vs prev cost %6.5f (learning rate %8.7f), avg seq cost %6.5f.\n", i+1, 100 * completion, avg_cost/sequence_counter, last_epoch_cost, LEARNING_RATE, avg_seq_cost / training_iterations);
				printf("%lu character sample from lstm below:\n", 10*training_iterations);
#endif

				printf("\n");
				for(int i = 3; i > 0; i--){
					printf("Sampling from lstm in %d\r", i);
					sleep(1);
				}
				sample(n, SAMPLE_CHARS, '\n');

				if(isnan(avg_cost) || avg_seq_cost/training_iterations > avg_cost/sequence_counter){
					printf("\nWARNING: average sequence cost was HIGHER than epoch average - something is probably wrong!\n");
				}else{
					printf("\nautosaving '%s'\n", modelfile);
					save_lstm(n, modelfile);
				}
				printf("\n***\nResuming training...\n");
				avg_seq_cost = 0;
				sleep(1);
				avg_seq_cost = 0;
#if STATEFUL
				wipe(n);
#endif
			}
			free(seq);
			seq = get_sequence(fp, &n->seq_len);
		}
		while(seq && n->seq_len > 0);
		fclose(fp);
		last_epoch_cost = avg_cost / sequence_counter;
	}
}

int main(int argc, char** argv){

	if(argc < 4){ printf("must provide %d args.\nexample usage: ./char [new/load] [path_to_modelfile] [path_to_datafile]\n", 3); exit(1);}

	printf("   _____ ____________ __ _   ______________\n");
	printf("  / ___//  _/ ____/ //_// | / / ____/_	__/\n");
	printf("  \\__ \\ / // __/ / ,<  /  |/ / __/   / /   \n");
	printf(" ___/ // // /___/ /| |/ /|  / /___  / /		 \n");
	printf("/____/___/_____/_/ |_/_/ |_/_____/ /_/		 \n");
	printf("																					 \n");
	printf("char-nn recurrent neural network interface.\n");

	srand(time(NULL));
	setbuf(stdout, NULL);

	bool newlstm;
	if(!strcmp(argv[1], "load")) newlstm = 0;
	else if(!strcmp(argv[1], "new")) newlstm = 1;
	else bad_args(argv[1], 0);
	
	char *modelfile = argv[2];
	char *datafile = argv[3];
	FILE *fp;

	LSTM n;
	if(newlstm){
		n = create_lstm(ASCII_RANGE, HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE, ASCII_RANGE);
		printf("creating '%s'\n", modelfile);
	}else{
		printf("loading '%s'\n", modelfile);
		fp = fopen(modelfile, "rb");
		if(!fp){ printf("Could not open modelfile '%s' - does it exist?\n", modelfile); exit(1);}
		n = load_lstm(modelfile);
		fclose(fp);
	}
	printf("network has %lu params.\n", n.num_params);
	save_lstm(&n, modelfile);

	if(!strcmp(datafile, "sample")){
		printf("Sampling from model '%s' below:\n", modelfile);
		sample(&n, SAMPLE_CHARS, '\n');
		exit(0);
	}

	fp = fopen(datafile, "rb");
	if(!fp){ printf("Could not open datafile '%s' - does it exist?\n", datafile); exit(1);}
	fclose(fp);

	train(&n, modelfile, datafile, NUM_EPOCHS, LEARNING_RATE);
	printf("training finished! LSTM saved to '%s'\n", modelfile);

}
