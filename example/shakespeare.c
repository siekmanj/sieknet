#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include "lstm.h"


#define CREATEONEHOT(name, size, index) float name[size]; memset(name, '\0', size*sizeof(float)); name[index] = 1.0;

/*
 * This LSTM is trained character-by-character on Shakespeare's complete works.
 * The end goal is that the network writes its own plays.
 */
 
//Calling convention: ../bin/shakespeare [load/new] [path_to_modelfile] [path_to_txt_file]

char *alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789,.;:?!'\"[]{}<>/-_*&|\\\n ";

char *modelfile = "../model/shakespeare_50.lstm";

char *datafile = "../data/shakespeare/complete_works.txt";


/*
 * Description: This function turns a character into a label using that character's position in an alphabet.
 * inpt: the character whose position in alphabet will be used to return a label.
 * alphabet: the list of all possible inputs and outputs of the network (needs to be the same length as input/output layer of network)
 */
int label_from_char(char inpt, const char *alphabet){
	for(int i = 0; i < strlen(alphabet); i++){
		if(alphabet[i] == inpt) return i;
	}
	if(inpt == EOF) return EOF;
	return strlen(alphabet)-1;
}

int main(void){
	srand(time(NULL));
	setbuf(stdout, NULL);

	printf("Press <ENTER> to load %s (may take a while to load), or enter <n> to create %s.\n", modelfile, modelfile);
	LSTM n;
	if(getchar() == 'n'){
		printf("creating network %s...\n", modelfile);
		n = create_lstm(strlen(alphabet), 40, strlen(alphabet));//loadLSTMFromFile(modelfile);
	}else{
		printf("loading %s...\n", modelfile);
		n = load_lstm(modelfile);
	}
	
	n.learning_rate = 0.01; //I've found that the larger the network, the lower the initial learning rate should be.	
	n.seq_len = 25;
	n.stateful = 1;
	int epochs = 5;
  float previousepochavgcost = 2.4;
	printf("Ready to train! Press <ENTER> to continue.\n");
	getchar();
	for(int i = 0; i < epochs; i++){ //Run for a large number of epochs
		printf("beginning epoch %d\n", i);
		FILE *fp = fopen(datafile, "rb"); //This is the dataset
		if(!fp){
			printf("%s couldn't be opened - does it exist?\n", datafile);
			exit(1);
		}
		size_t datafilelen = 5447092;
	
		int input_character = label_from_char(' ', alphabet); //Start the network off with a character
		int label = label_from_char(fgetc(fp), alphabet); //Get the first letter from the dataset as a label

		float cost = 0;
		float lastcost = 0;
		int count = 0;
		float lastavgcost = previousepochavgcost;
		size_t epochcount = 1;
		float epochcost = 0;
		do {
			//The below is all the code needed for training - the rest is just debug stuff.
			/****************************************************/
			CREATEONEHOT(x, strlen(alphabet), input_character);
			CREATEONEHOT(y, strlen(alphabet), label);

			lstm_forward(&n, x);
			float cost_local = n.cost(&n, y);
			lstm_backward(&n);
			
			/****************************************************/
			if(isnan(cost_local)) { printf("COST NAN! STOPPING...\n"); exit(1); }

			cost += cost_local;

			int guess = n.guess;

			if(alphabet[label] == '\n') printf("\n");
			else if(alphabet[guess] == alphabet[label]) printf("%c", alphabet[label]);
			else printf("_");
//			printf("%c", alphabet[guess]);

			input_character = label;
			count++;
			label = label_from_char(fgetc(fp), alphabet);
			
			int sequences = 500;
			if(count % (n.seq_len*sequences) == 0){
				int seed = input_character;
				printf("\n%d TRAINING CYCLES DONE, PRINTING SAMPLE BELOW FROM '%c'\n", sequences, seed);
				for(int k = 0; k < n.seq_len*10; k++){
					CREATEONEHOT(tmp, strlen(alphabet), seed);
					lstm_forward(&n, tmp);
					int guess = n.output_layer.guess;
					printf("%c", alphabet[guess]);
					seed = guess;
				}
				printf("\n");

				epochcount += count;
				epochcost += cost;
			
				//Debug stuff
				float completion = 100 * (float)epochcount/datafilelen;
				if(lastcost == 0) lastcost = cost/count;
				lastcost = (lastcost * 10 + cost/count)/11.0;
				printf("\n\n****\nlatest cost: %6.5f (avg %6.5f) vs epoch avg cost:%f, epoch (%d) %5.2f%% completed.\n", cost/count, lastcost, epochcost/epochcount, i, completion);

				if(cost/count < epochcost/epochcount){
					printf("\nAUTOSAVING MODEL FILE!\n");
					save_lstm(&n, modelfile);
				}else{
					printf("\nPERFORMANCE WORSE THAN AVERAGE, NOT SAVING MODELFILE\n");
				}
				printf("****\n\n");

				//Reset short-term cost statistic
				cost = 0;
				count = 0;
				wipe(&n);
				sleep(1);
			}
		}
	
		while(label != EOF);
		fclose(fp);
		printf("Epoch completed, cost was %f vs previous cost of %f\n", cost/count, lastavgcost);
		previousepochavgcost = epochcost/epochcount;
		if(!(i%10)) getchar();

	}
}
