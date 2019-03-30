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

#define USE_MOMENTUM 1

#define CHAR_OUTPUT 0

typedef uint8_t bool;

size_t HIDDEN_LAYER_SIZE = 512;
size_t NUM_EPOCHS				 = 10;
size_t SEQ_LEN					 = 75;
size_t ASCII_RANGE			 = 96; //96 useful characters in ascii: A-Z, a-z, 0-9, !@#$%...etc
size_t SAMPLE_EVERY			 = 100;
size_t SAMPLE_CHARS			 = 1000;

float LEARNING_RATE			 = 0.0001;
float MOMENTUM					 = 0.95;

/*
 * This file is for training an LSTM character-by-character on any text (ascii) file provided.
 */

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
	char *ret = (char*)malloc(*size*sizeof(char));
	for(int i = 0; i < *size; i++){
		ret[i] = fgetc(fp);
		if(ret[i] == EOF){
			*size = 0;
			return NULL;
		}
	}
	return ret;
}

struct timespec diff(struct timespec start, struct timespec end){
	struct timespec temp;
	if ((end.tv_nsec-start.tv_nsec)<0){
		temp.tv_sec = end.tv_sec-start.tv_sec-1;
		temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
	}else{
		temp.tv_sec = end.tv_sec-start.tv_sec;
		temp.tv_nsec = end.tv_nsec-start.tv_nsec;
	}
	return temp;
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

	Momentum o = create_optimizer(Momentum, *n);
	o.alpha = learning_rate;
	o.beta = MOMENTUM;

	FILE *tmp = fopen(datafile, "rb");
	fseek(tmp, 0, SEEK_END);
	size_t datafilelen = ftell(tmp);
	fclose(tmp);

	float learning_schedule[] = {
															 learning_rate * 1.0,
															 learning_rate * 0.7, 
															 learning_rate * 0.5, 
															 learning_rate * 0.5, 
															 learning_rate * 0.25, 
															 learning_rate * 0.125, 
															 learning_rate * 0.1,
															 learning_rate * 0.1,
															 learning_rate * 0.1,
															 learning_rate * 0.05,
															 learning_rate * 0.05,
															 learning_rate * 0.05
															};
	float last_epoch_cost = 4.5;
	for(int i = 0; i < num_epochs; i++){

		n->seq_len	= SEQ_LEN;
		n->stateful = 1;
		o.alpha = learning_schedule[i];

		FILE *fp = fopen(datafile, "rb");
		fseek(fp, 0, SEEK_SET);

		size_t training_iterations = SAMPLE_EVERY;
		size_t sequence_counter = 0;

		size_t ctr = 0;

		float avg_cost = 0;
		float avg_seq_cost = 0;
		float seq_time = 0;
		float avg_seq_time = 0;

		wipe(n);

		char *seq = get_sequence(fp, &n->seq_len);
		char input_char = '\n';
		do{
			struct timespec start, end;
			clock_gettime(CLOCK_REALTIME, &start);

			float completion = ((float)ctr/datafilelen);
			float time_left = (1-completion) * (datafilelen / n->seq_len) * ((avg_seq_time / (sequence_counter % training_iterations + 1)));
			int hrs_left = (int)(time_left / (60*60));
			int min_left = ((int)(time_left - (hrs_left * 60 * 60))) / 60;
			//printf("%5.4fs, appr. %2dh %2dm left.\n", seq_time, hrs_left, min_left);
			printf("seqlen %3lu | (%3lu)/(%3lu) | (%5.4f s, appr. %2dh %2dmin left) | epoch %2d %4.2f%% | last %3lu seqs: %4.3f | epoch cost: %5.4f | previous epoch: %5.4f | lr: %7.6f\r",
						n->seq_len, 
						sequence_counter % training_iterations + 1 , 
						training_iterations,
						seq_time,
						hrs_left,
						min_left,
						i,
						100 * completion, 
						sequence_counter % training_iterations + 1, 
						avg_seq_cost / (sequence_counter % training_iterations + 1), 
						avg_cost/sequence_counter, 
						last_epoch_cost,
						o.alpha
					 );
			float seq_cost = 0;
			for(int j = 0; j < n->seq_len; j++){
				char label = seq[j];

				CREATEONEHOT(x, ASCII_RANGE, char2int(input_char));
				CREATEONEHOT(y, ASCII_RANGE, char2int(label));
				
				lstm_forward(n, x);
				float c = lstm_cost(n, y);
				lstm_backward(n);

				if(!n->t) o.step(o);

				seq_cost += c;
				ctr++;

				input_char = label;
			}
			avg_seq_cost += seq_cost / n->seq_len;
			avg_cost += seq_cost / n->seq_len;
			sequence_counter++;

			if(!(sequence_counter % (training_iterations))){
				printf("\n");
				wipe(n);
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
				avg_seq_time = 0;
				wipe(n);
			}
			free(seq);
			seq = get_sequence(fp, &n->seq_len);
			clock_gettime(CLOCK_REALTIME, &end);
			struct timespec elapsed = diff(start, end);
			seq_time = (double)elapsed.tv_sec + ((double)elapsed.tv_nsec) / 1000000000;
			avg_seq_time += seq_time;
		}
		while(seq && n->seq_len > 0);
		fclose(fp);
		last_epoch_cost = avg_cost / sequence_counter;
	}
}

int main(int argc, char** argv){

	printf("made it to %d\n", __LINE__);
	if(argc < 4){ printf("%d args needed. Usage: ./char [new/load] [path_to_modelfile] [path_to_datafile]\n", 3); exit(1);}

	printf("   _____ ____________ __ _   ______________\n");
	printf("  / ___//  _/ ____/ //_// | / / ____/_  __/\n");
	printf("  \\__ \\ / // __/ / ,<  /  |/ / __/   / /   \n");
	printf(" ___/ // // /___/ /| |/ /|  / /___  / /    \n");
	printf("/____/___/_____/_/ |_/_/ |_/_____/ /_/     \n");
	printf("																					 \n");
	printf("ascii-nn recurrent neural network interface.\n");

	//srand(time(NULL));
	srand(1);
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
		n = create_lstm(ASCII_RANGE, HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE, ASCII_RANGE);
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
